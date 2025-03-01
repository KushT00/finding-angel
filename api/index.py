from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth, firestore
import google.generativeai as genai
import os
from dotenv import load_dotenv
import openai
import csv
import tempfile
from datetime import datetime
from io import StringIO
from flask import send_file
import threading
import razorpay
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Load environment variables from .env file
load_dotenv()
# Update CORS configuration
CORS(app, 
     resources={r"/api/*": {
         "origins": ["https://findmyangel.vercel.app", "http://127.0.0.1:3000"],  # Add both localhost variants
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "expose_headers": ["Content-Type", "Authorization"]
     }})

# Add OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', 'https://findmyangel.vercel.app')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Authorization, Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        
        return response




@app.route('/test', methods=['GET'])
def hello():
    try:
        return jsonify({"message": "Hello from FindMyAngel API!!!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize Firebase Admin if not already initialized

firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")

if firebase_credentials:
    try:
        cred_dict = json.loads(firebase_credentials)
        cred = credentials.Certificate(cred_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized successfully")
    except Exception as e:
        logger.error(f"Firebase Admin initialization error: {str(e)}")
        raise e
else:
    logger.error("FIREBASE_CREDENTIALS not found in .env file")
    raise Exception("FIREBASE_CREDENTIALS not set")

# Initialize Razorpay client
razorpay_client = razorpay.Client(
    auth=(os.getenv('RAZORPAY_KEY_ID'), os.getenv('RAZORPAY_KEY_SECRET'))
)



# Initialize Firestore
db = firestore.client()

# Configure Gemini API with the key from .env
api_key = "AIzaSyCx0yl-9ToWcveCx1D8J20cVr-6pIbFFOM"
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
    
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Add this at the top of your file with other imports
INVESTOR_TYPE_MAPPINGS = {
    'venture_fund': ['venture fund', 'vc', 'corporate vc'],
    'angel': ['angel', 'angel network'],
    'accelerator': ['accelerator'],
    'family_office': ['family office'],
    'corporate': ['corporate vc']
}

# Add a lock for thread safety
credit_lock = threading.Lock()

# Add safety settings to allow more content
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "block_none",
    "HARM_CATEGORY_HATE_SPEECH": "block_none",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
}

def verify_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.debug("Starting token verification")
        try:
            auth_header = request.headers.get('Authorization', '')
            logger.debug(f"Auth header: {auth_header[:20]}...")  # Log first 20 chars

            if not auth_header.startswith('Bearer '):
                logger.error("Missing or invalid Bearer token")
                return jsonify({'error': 'Invalid token format'}), 401

            token = auth_header.split('Bearer ')[1].strip()
            logger.debug(f"Extracted token: {token[:20]}...")  # Log first 20 chars

            try:
                # Print Firebase Admin credentials status
                logger.debug(f"Firebase Admin App: {firebase_admin._apps}")
                
                decoded_token = auth.verify_id_token(token)
                logger.info(f"Token verified for user: {decoded_token.get('uid')}")
                request.user = decoded_token
                return f(*args, **kwargs)
            except auth.InvalidIdTokenError as e:
                logger.error(f"Invalid token: {str(e)}")
                return jsonify({'error': 'Invalid token'}), 401
            except Exception as e:
                logger.error(f"Token verification error: {str(e)}")
                return jsonify({'error': str(e)}), 401

        except Exception as e:
            logger.error(f"Decorator error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return decorated_function

def load_data():
    """Load and transform JSON data correctly"""
    with open("temp.json", "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    structured_data = []
    for investor in raw_data:
        details = investor.get("cellValuesByColumnId", {})
        
        # Debug: Print Fund_Type value
        fund_type = details.get("Fund_Type", "")
        # print(f"Processing investor with Fund_Type: {fund_type}")
        
        structured_data.append({
            "id": investor.get("id", ""),
            "name": details.get("Name", ""),
            "headline": details.get("Fund_Desc", ""),
            "company_name": details.get("Investor_Name", ""),
            "website": details.get("website", ""),
            "stage": details.get("fund_Stage", []),
            "fund_focus": details.get("Fund_Focus", []),
            "email": details.get("email", ""),
            "location": details.get("location", []),
            "Fund_Type": details.get("Fund_Type", ""),  # Make sure this matches your data structure
            "No_Of_Exits": details.get("No_Of_Exits", ""),
            "No_Of_Investments": details.get("No_Of_Investments", ""),
            "Founding_Year": details.get("Founding_Year", ""),
            "Linkedin": details.get("Linkedin", ""),
            "twitter": details.get("twitter", ""),
            "Facebook_Link": details.get("Facebook_Link", ""),
            "portfolio_Companies": details.get("portfolio_Companies", [])
        })
    
    return structured_data

@app.route("/api/investors", methods=["GET"])
@verify_token
def get_investors():
    try:
        user_id = request.user['uid']
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
            
        user_credits = user_doc.to_dict().get('credits', 0)
        
        search = request.args.get('search', '').lower().strip()
        location = request.args.get('location', '')
        industry = request.args.get('industry', '')
        
        investors = load_data()
        filtered_investors = []

        def redact_text(text):
            if not text:
                return ''
            if user_credits > 0:
                return text
            # Keep first letter of each word, replace rest with X
            words = text.split()
            redacted = []
            for word in words:
                if len(word) <= 1:
                    redacted.append(word)
                else:
                    redacted.append(word[0] + 'X' * (len(word) - 1))
            return ' '.join(redacted)
        
        for investor in investors:
            include_investor = True
            
            # Apply filters...
            if location and location not in investor.get('location', []):
                include_investor = False
                continue
                
            if industry:
                investor_focus = [focus.lower().strip() for focus in investor.get('fund_focus', [])]
                if industry.lower().strip() not in investor_focus:
                    include_investor = False
                    continue
            
            if search:
                searchable_text = ' '.join([
                    str(investor.get('name', '')).lower(),
                    str(investor.get('company_name', '')).lower(),
                    str(investor.get('headline', '')).lower(),
                    ' '.join(str(focus).lower() for focus in investor.get('fund_focus', [])),
                    ' '.join(str(loc).lower() for loc in investor.get('location', [])),
                ])
                
                if not all(term in searchable_text for term in search.split()):
                    include_investor = False
                    continue
            
            if include_investor:
                # Redact data if user has no credits
                filtered_investor = {
                    'id': investor.get('id', ''),
                    'name': redact_text(investor.get('name', '')),
                    'headline': redact_text(investor.get('headline', '')),
                    'company_name': redact_text(investor.get('company_name', '')),
                    'location': investor.get('location', []),  # Keep locations visible
                    'fund_focus': investor.get('fund_focus', [])  # Keep fund focus visible
                }
                filtered_investors.append(filtered_investor)

        logger.info(f"Returning {len(filtered_investors)} investors with {'redacted' if user_credits <= 0 else 'full'} data")
        return jsonify(filtered_investors)

    except Exception as e:
        logger.error(f"Error in get_investors: {str(e)}")
        return jsonify({"error": str(e)}), 500

def has_sufficient_credits(user_id):
    """Check if user has credits without deducting them"""
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        return False
        
    user_data = user_doc.to_dict()
    return user_data.get('credits', 0) > 0

@app.route("/api/investors/<investor_id>", methods=["GET"])
@verify_token
def get_investor(investor_id):
    try:
        user_id = request.user['uid']
        
        # Check and deduct credits here
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
            
        current_credits = user_doc.to_dict().get('credits', 0)
        if current_credits <= 0:
            return jsonify({"error": "Insufficient credits"}), 403
        
        investors = load_data()
        investor = next(
            (inv for inv in investors if inv.get('id') == investor_id),
            None
        )
        
        if not investor:
            return jsonify({"error": "Investor not found"}), 404

        # This is now the only place where we deduct credits
        with credit_lock:
            user_ref.update({
                'credits': current_credits - 1
            })

        return jsonify({
            "investor": investor,
            "remainingCredits": current_credits - 1
        })
        
    except Exception as e:
        print(f"Error in get_investor: {str(e)}")
        return jsonify({"error": "An error occurred while fetching investor"}), 500

@app.route("/api/investors/stats", methods=["GET"])
@verify_token
def get_stats():
    """Get investor database statistics"""
    investors = load_data()
    
    # Calculate statistics
    total_investors = len(investors)
    unique_industries = set()
    locations = set()
    
    for investor in investors:
        unique_industries.update(investor["fund_focus"])
        locations.update(investor["location"])
    
    return jsonify({
        "total_investors": total_investors,
        "total_industries": len(unique_industries),
        "total_locations": len(locations),
        "industries": list(unique_industries),
        "locations": list(locations)
    })

@app.route("/api/investors/advanced", methods=["POST"])
@verify_token
def advanced_filter():
    """Advanced filtering for investors"""
    try:
        user_id = request.user['uid']
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
            
        user_credits = user_doc.to_dict().get('credits', 0)
        
        investors = load_data()
        filters = request.json
        logger.debug("Received filters: %s", filters)
        
        def redact_text(text):
            if not text:
                return ''
            if user_credits > 0:
                return text
            words = text.split()
            redacted = []
            for word in words:
                if len(word) <= 1:
                    redacted.append(word)
                else:
                    redacted.append(word[0] + 'X' * (len(word) - 1))
            return ' '.join(redacted)

        filtered_investors = []
        for investor in investors:
            matches = True
            
            # Investment Type with mapping
            if filters.get('fund_types') and len(filters['fund_types']) > 0:
                investor_type = str(investor.get('Fund_Type', '')).strip().lower()
                matches_type = False
                for fund_type in filters['fund_types']:
                    if investor_type in INVESTOR_TYPE_MAPPINGS.get(fund_type, []):
                        matches_type = True
                        break
                if not matches_type:
                    matches = False
                    continue

            # Location filter
            if filters.get('locations') and len(filters['locations']) > 0:
                if not any(loc in investor.get('location', []) for loc in filters['locations']):
                    matches = False
                    continue

            # Industry/Focus filter
            if filters.get('industries') and len(filters['industries']) > 0:
                if not any(industry in investor.get('fund_focus', []) for industry in filters['industries']):
                    matches = False
                    continue

            if matches:
                # Only include required fields with redaction if needed
                filtered_investor = {
                    'id': investor.get('id', ''),
                    'name': redact_text(investor.get('name', '')),
                    'headline': redact_text(investor.get('headline', '')),
                    'company_name': redact_text(investor.get('company_name', '')),
                    'location': investor.get('location', []),
                    'fund_focus': investor.get('fund_focus', [])
                }
                filtered_investors.append(filtered_investor)

        logger.info(f"Advanced search returning {len(filtered_investors)} investors with {'redacted' if user_credits <= 0 else 'full'} data")
        return jsonify(filtered_investors)

    except Exception as e:
        logger.error(f"Error in advanced_filter: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/investors/<investor_id>/description", methods=["GET"])
@verify_token
def get_investor_description(investor_id):
    try:
        user_id = request.user['uid']
        
        # Only check credits, don't deduct
        if not has_sufficient_credits(user_id):
            return jsonify({"error": "Insufficient credits"}), 403
            
        # Rest of the description logic without credit deduction
        investors = load_data()
        investor = next(
            (inv for inv in investors if inv.get('id') == investor_id),
            None
        )
        
        if not investor:
            return jsonify({"error": "Investor not found"}), 404
            
        return jsonify({
            "description": investor.get('headline', ''),
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error getting description:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/api/investors/<investor_id>/ai-summary", methods=['GET'])
@verify_token
def get_investor_ai_summary(investor_id):
    try:
        user_id = request.user['uid']
        
        # Only check credits without deducting
        if not has_sufficient_credits(user_id):
            return jsonify({"error": "Insufficient credits"}), 403

        # Get investor data
        investors = load_data()
        investor = next((inv for inv in investors if inv.get('id') == investor_id), None)
        
        if not investor:
            return jsonify({"error": "Investor not found"}), 404

        # Generate AI summary
        summary_prompt = f"""
        Create a 9-10 sentence summary of this investor:
        
        Name: {investor.get('name')}
        Type: {investor.get('Fund_Type')}
        Focus Areas: {', '.join(investor.get('fund_focus', []))}
        Investment Stages: {', '.join(investor.get('stage', []))}
        Location: {', '.join(investor.get('location', []))}
        Investments: {investor.get('No_Of_Investments')}
        Exits: {investor.get('No_Of_Exits')}
        Description: {investor.get('headline')}
        """

        response = model.generate_content(
            summary_prompt,
            safety_settings=safety_settings
        )
        
        ai_summary = response.text.strip()

        # Get current credits for response
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        current_credits = user_doc.to_dict().get('credits', 0)

        return jsonify({
            "ai_summary": ai_summary,
            "status": "success",
            "remainingCredits": current_credits
        })

    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

def mask_text(text, preserve_first_word=False):
    """
    Mask text while keeping first letter of each word and replacing rest with X
    If preserve_first_word is True, keep the first word intact
    """
    if not text:
        return ""
        
    if preserve_first_word:
        words = text.split()
        if not words:
            return ""
        
        # Keep first word, mask the rest
        first_word = words[0]
        masked_words = [first_word] + [w[0] + "X" * (len(w) - 1) if len(w) > 1 else w for w in words[1:]]
        return " ".join(masked_words)
    else:
        # Mask all words
        words = text.split()
        masked_words = [w[0] + "X" * (len(w) - 1) if len(w) > 1 else w for w in words]
        return " ".join(masked_words)

def strip_investor_data(investor):
    """Create a stripped version of investor data with masked fields"""
    return {
        "id": investor.get("id", ""),
        "name": mask_text(investor.get("name", "")),
        "headline": mask_text(investor.get("headline", ""), preserve_first_word=True),
        "company_name": mask_text(investor.get("company_name", "")),
        "fund_focus": investor.get("fund_focus", []),
        "location": investor.get("location", []),
        # Remove sensitive data
        "email": "",
        "website": "",
        "Linkedin": "",
        "twitter": "",
        "Facebook_Link": "",
        # Mask numbers
        "No_Of_Investments": "XX",
        "No_Of_Exits": "XX",
        "Founding_Year": "XXXX",
        # Mask portfolio companies names
        "portfolio_Companies": [
            {"foreignRowDisplayName": mask_text(company.get("foreignRowDisplayName", ""))}
            for company in investor.get("portfolio_Companies", [])
        ]
    }

@app.route("/api/investors/stripped", methods=["GET"])
@verify_token
def get_stripped_investors():
    try:
        search = request.args.get('search', '').lower().strip()
        location = request.args.get('location', '')
        industry = request.args.get('industry', '')
        
        investors = load_data()
        filtered_investors = []
        
        for investor in investors:
            include_investor = True
            
            # Apply existing filters
            if location and location not in investor.get('location', []):
                include_investor = False
                continue
                
            if industry:
                investor_focus = [focus.lower().strip() for focus in investor.get('fund_focus', [])]
                if industry.lower().strip() not in investor_focus:
                    include_investor = False
                    continue
            
            if search:
                searchable_text = ' '.join([
                    str(investor.get('name', '')).lower(),
                    str(investor.get('company_name', '')).lower(),
                    str(investor.get('headline', '')).lower(),
                    ' '.join(str(focus).lower() for focus in investor.get('fund_focus', [])),
                    ' '.join(str(loc).lower() for loc in investor.get('location', [])),
                    str(investor.get('Fund_Type', '')).lower()
                ])
                
                if not all(term in searchable_text for term in search.split()):
                    include_investor = False
                    continue
            
            if include_investor:
                # Strip sensitive data before adding to results
                filtered_investors.append(strip_investor_data(investor))

        return jsonify(filtered_investors)

    except Exception as e:
        print(f"Error in get_stripped_investors: {str(e)}")
        return jsonify({"error": "An error occurred while fetching investors"}), 500

@app.route("/api/investors/stripped/<investor_id>", methods=["GET"])
@verify_token
def get_stripped_investor(investor_id):
    try:
        investors = load_data()
        investor = next((inv for inv in investors if inv.get('id') == investor_id), None)
        
        if not investor:
            return jsonify({"error": "Investor not found"}), 404
            
        # Return stripped version of investor data without deducting credits
        return jsonify(strip_investor_data(investor))
        
    except Exception as e:
        print(f"Error fetching stripped investor {investor_id}: {str(e)}")
        return jsonify({"error": "An error occurred while fetching investor"}), 500

@app.route("/api/investors/<investor_id>/download", methods=["GET"])
@verify_token
def download_investor_profile(investor_id):
    try:
        investors = load_data()
        # Clean the investor_id by removing any URL parts
        clean_id = investor_id.replace('http://', '').replace('https://', '').replace('www.', '')
        investor = next((inv for inv in investors if inv.get('id') == clean_id), None)
        
        if not investor:
            # Try matching by website
            investor = next((inv for inv in investors 
                           if inv.get('website', '').replace('http://', '')
                              .replace('https://', '').replace('www.', '') == clean_id), None)
        
        if not investor:
            return jsonify({"error": "Investor not found"}), 404

        # Helper function to format website URL
        def format_website_url(url):
            if not url:
                return ''
            url = url.replace('http://', '').replace('https://', '').replace('www.', '')
            return f'https://www.{url}'

        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow([f"{investor.get('name', 'Investor')} Profile"])
        writer.writerow(["Name", investor.get('name', '')])
        writer.writerow(["Company", investor.get('company_name', '')])
        writer.writerow(["Fund Type", investor.get('Fund_Type', '')])
        writer.writerow(["Founded", investor.get('Founding_Year', '')])
        writer.writerow(["Description", investor.get('headline', '')])
        writer.writerow(["Total Investments", investor.get('No_Of_Investments', '0')])
        writer.writerow(["Total Exits", investor.get('No_Of_Exits', '0')])
        writer.writerow(["Investment Stages", ", ".join(investor.get('stage', []))])
        writer.writerow(["Industries", ", ".join(investor.get('fund_focus', []))])
        writer.writerow(["Locations", ", ".join(investor.get('location', []))])
        
        if investor.get('portfolio_Companies'):
            writer.writerow(["Portfolio Companies"])
            for company in investor.get('portfolio_Companies', []):
                if isinstance(company, dict) and 'foreignRowDisplayName' in company:
                    writer.writerow(["", company['foreignRowDisplayName'].strip()])
        
        writer.writerow(["Email", investor.get('email', '')])
        writer.writerow(["Website", format_website_url(investor.get('website', ''))])
        writer.writerow(["LinkedIn", format_website_url(investor.get('Linkedin', ''))])
        writer.writerow(["Twitter", format_website_url(investor.get('twitter', ''))])
        writer.writerow(["Facebook", format_website_url(investor.get('Facebook_Link', ''))])
        writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Source", "FindMyAngel Platform"])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_file.write(output.getvalue())
            temp_file_path = temp_file.name
            
        return send_file(
            temp_file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{investor.get('name', 'investor')}_profile_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
    except Exception as e:
        print(f"Error downloading investor profile: {str(e)}")
        return jsonify({"error": "An error occurred while downloading profile"}), 500

@app.route("/api/create-order", methods=["POST"])
@verify_token
def create_order():
    logger.debug("Starting create_order")
    try:
        # Log the full request
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Request data: {request.get_data()}")
        
        user_id = request.user.get('uid')
        logger.debug(f"User ID from token: {user_id}")

        if not user_id:
            logger.error("No user ID found in token")
            return jsonify({"error": "User not found"}), 401

        data = request.json
        logger.debug(f"Parsed request data: {data}")

        if not data:
            logger.error("No request data found")
            return jsonify({"error": "No request data"}), 400

        amount = data.get('amount')
        credits = data.get('credits')
        plan_id = data.get('planId')

        logger.debug(f"Order details - Amount: {amount}, Credits: {credits}, Plan ID: {plan_id}")

        if not all([amount, credits, plan_id]):
            logger.error("Missing required fields")
            return jsonify({"error": "Missing required fields"}), 400

        # Log Razorpay client status
        logger.debug(f"Razorpay client: {razorpay_client}")

        order_data = {
            'amount': int(amount * 100),
            'currency': 'INR',
            'receipt': f'order_{datetime.now().timestamp()}',
            'notes': {
                'user_id': user_id,
                'credits': credits,
                'plan_id': plan_id
            }
        }
        
        logger.debug(f"Creating Razorpay order with data: {order_data}")

        try:
            order = razorpay_client.order.create(data=order_data)
            logger.info(f"Razorpay order created: {order}")
            return jsonify(order)
        except Exception as e:
            logger.error(f"Razorpay order creation error: {str(e)}")
            return jsonify({"error": f"Failed to create Razorpay order: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Create order error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add route for verifying payment
@app.route("/api/verify-payment", methods=["POST"])
@verify_token
def verify_payment():
    try:
        logger.debug("Starting payment verification")
        data = request.json
        user_id = request.user['uid']
        
        logger.debug(f"Payment data received: {data}")
        logger.debug(f"User ID: {user_id}")
        
        # Verify payment signature
        params_dict = {
            'razorpay_payment_id': data['razorpay_payment_id'],
            'razorpay_order_id': data['razorpay_order_id'],
            'razorpay_signature': data['razorpay_signature']
        }
        
        try:
            razorpay_client.utility.verify_payment_signature(params_dict)
            logger.info("Payment signature verified successfully")
        except Exception as e:
            logger.error(f"Payment signature verification failed: {str(e)}")
            return jsonify({"error": "Invalid payment signature"}), 400
        
        # Get the order details to verify amount and credits
        try:
            order = razorpay_client.order.fetch(data['razorpay_order_id'])
            logger.debug(f"Order details fetched: {order}")
        except Exception as e:
            logger.error(f"Failed to fetch order details: {str(e)}")
            return jsonify({"error": "Failed to verify order details"}), 400
        
        # Update user credits in Firestore
        user_ref = db.collection('users').document(user_id)
        
        # Use transaction to ensure atomic update
        @firestore.transactional
        def update_credits_transaction(transaction, user_ref, credits_to_add):
            user_doc = user_ref.get(transaction=transaction)
            if not user_doc.exists:
                raise ValueError("User not found")
                
            user_data = user_doc.to_dict()
            current_credits = user_data.get('credits', 0)
            new_credits = current_credits + credits_to_add
            
            # Update user document
            transaction.update(user_ref, {
                'credits': new_credits,
                'last_payment': firestore.SERVER_TIMESTAMP,
                'payment_history': firestore.ArrayUnion([{
                    'payment_id': data['razorpay_payment_id'],
                    'order_id': data['razorpay_order_id'],
                    'amount': order['amount'] / 100,  # Convert from paise to rupees
                    'credits': credits_to_add,
                    'timestamp': firestore.SERVER_TIMESTAMP
                }])
            })
            
            return new_credits

        try:
            transaction = db.transaction()
            credits_to_add = int(order['notes']['credits'])
            new_credits = update_credits_transaction(transaction, user_ref, credits_to_add)
            
            logger.info(f"Credits updated successfully. New balance: {new_credits}")
            
            return jsonify({
                "status": "success",
                "message": "Payment verified and credits updated",
                "credits": new_credits
            })
            
        except Exception as e:
            logger.error(f"Failed to update credits: {str(e)}")
            return jsonify({"error": "Failed to update credits"}), 500
            
    except Exception as e:
        logger.error(f"Payment verification error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Add new route for community members
@app.route("/api/community/join", methods=["POST"])
@verify_token
def join_community():
    try:
        # Log incoming request
        logger.debug("Received community join request")
        logger.debug(f"Headers: {dict(request.headers)}")
        
        user_id = request.user['uid']
        data = request.json
        
        logger.debug(f"Request data: {data}")

        # Validate required fields
        required_fields = ['name', 'email', 'startupName', 'location']
        for field in required_fields:
            if not data.get(field):
                logger.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        try:
            # Create community member document
            community_ref = db.collection('community_members').document()
            
            community_data = {
                'userId': user_id,
                'name': data['name'],
                'email': data['email'],
                'startupName': data['startupName'],
                'location': data['location'],
                'socialMediaLink': data.get('socialMediaLink', ''),
                'createdAt': firestore.SERVER_TIMESTAMP,
                'status': 'pending',
            }

            # Add the document to Firestore
            community_ref.set(community_data)
            
            logger.info(f"Successfully created community member with ID: {community_ref.id}")

            # Set CORS headers in the response
            response = jsonify({
                "status": "success",
                "message": "Successfully joined the community",
                "memberId": community_ref.id
            })
            
            response.headers.add('Access-Control-Allow-Origin', 'https://findmyangel.vercel.app')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response

        except Exception as e:
            logger.error(f"Firestore error: {str(e)}")
            return jsonify({
                "error": "Database error",
                "message": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error joining community: {str(e)}")
        return jsonify({
            "error": "Failed to join community",
            "message": str(e)
        }), 500

@app.route("/api/contact", methods=["POST"])
def submit_contact():
    try:
        logger.debug("Received contact form submission")
        data = request.json
        
        logger.debug(f"Contact form data: {data}")

        # Validate required fields
        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if not data.get(field):
                logger.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        try:
            # Create contact submission document
            contact_ref = db.collection('contact_submissions').document()
            
            contact_data = {
                'userId': 'anonymous',  # Since user might not be logged in
                'name': data['name'],
                'email': data['email'],
                'subject': data['subject'],
                'message': data['message'],
                'createdAt': firestore.SERVER_TIMESTAMP,
                'status': 'unread'
            }

            # Add the document to Firestore
            contact_ref.set(contact_data)
            
            logger.info(f"Successfully created contact submission with ID: {contact_ref.id}")

            response = jsonify({
                "status": "success",
                "message": "Message sent successfully",
                "submissionId": contact_ref.id
            })
            
            response.headers.add('Access-Control-Allow-Origin', 'https://findmyangel.vercel.app')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response

        except Exception as e:
            logger.error(f"Firestore error: {str(e)}")
            return jsonify({
                "error": "Database error",
                "message": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error submitting contact form: {str(e)}")
        return jsonify({
            "error": "Failed to submit message",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

# This is what Vercel will use to run your app
app.debug = False
