"use client";
import { useEffect, useState } from 'react';
import { auth } from '../../lib/firebase';
import { useRouter } from 'next/navigation';
import { Search, Moon, Sun, ChevronLeft, ChevronRight, Filter, X, Eye, Check, Bookmark } from 'lucide-react';
import { doc, getDoc, setDoc, Timestamp, collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../../lib/firebase';
import { useDarkMode } from '../../context/DarkModeContext';
import { toast } from 'react-hot-toast';

interface Investor {
  id: string;
  name: string;
  headline: string;
  company_name: string;
  location: string[];
  fund_focus: string[];
  website: string;
  email: string;
  twitter: string;
  linkedin: string;
  No_Of_Exits: string;
  No_Of_Investments: string;
  Fund_Type: string;
}

interface Stats {
  total_investors: number;
  total_industries: number;
  total_locations: number;
  industries: string[];
  locations: string[];
}

interface UserData {
  credits: number;
  username: string;
  createdAt: Timestamp;
  uuid: string;
}

interface UserCredits {
  total: number;
}

interface AdvancedFilters {
  stages: string[];
  fund_types: string[];
  investment_range: {
    min?: number;
    max?: number;
  };
  exits_range: {
    min?: number;
    max?: number;
  };
  industries: string[];
}

// Add new interface for community form
interface CommunityFormData {
  name: string;
  email: string;
  startupName: string;
  location: string;
  socialMediaLink: string;
}

const INVESTMENT_STAGES = [
  { id: 'selXQOMmYzE7ya8EG', name: 'Pre-Seed', color: 'blue' },
  { id: 'selvBkDYwQkK10drd', name: 'Seed', color: 'cyan' },
  { id: 'seljvGJzunjQlyjvM', name: 'Series A', color: 'teal' },
  { id: 'sel3PwWuYBL3P03bV', name: 'Series B', color: 'green' },
  { id: 'seliHJjZZu48Onn0R', name: 'Series C', color: 'yellow' },
  { id: 'selpN10Qz6CRaHb1c', name: 'Series D', color: 'orange' }
];

const INVESTOR_TYPES = [
  { value: 'venture_fund', label: 'Venture Fund/VC', matchTerms: ['venture fund', 'vc', 'corporate vc'] },
  { value: 'angel', label: 'Angel/Angel Network', matchTerms: ['angel', 'angel network'] },
  { value: 'accelerator', label: 'Accelerator', matchTerms: ['accelerator'] },
  { value: 'family_office', label: 'Family Office', matchTerms: ['family office'] },
  { value: 'corporate', label: 'Corporate VC', matchTerms: ['corporate vc'] }
];

const RAZORPAY_KEY = "rzp_test_KsZM1Alo8MufnE"; // Replace with your test key

export default function Dashboard() {
  const { darkMode, toggleDarkMode } = useDarkMode();
  const [user, setUser] = useState(null);
  const [investors, setInvestors] = useState<Investor[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [credits, setCredits] = useState<UserCredits>({ total: 0 });
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLocation, setSelectedLocation] = useState('');
  const [selectedIndustry, setSelectedIndustry] = useState('');
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [advancedFilters, setAdvancedFilters] = useState<AdvancedFilters>({
    stages: [],
    fund_types: [],
    investment_range: {},
    exits_range: {},
    industries: []
  });
  const [isSearching, setIsSearching] = useState(false);
  const [viewedInvestors, setViewedInvestors] = useState<Set<string>>(new Set());
  const [showCreditModal, setShowCreditModal] = useState(false);
  const [loadingPayment, setLoadingPayment] = useState(false);
  const [showCommunityModal, setShowCommunityModal] = useState(false);
  const [communityForm, setCommunityForm] = useState<CommunityFormData>({
    name: '',
    email: '',
    startupName: '',
    location: '',
    socialMediaLink: ''
  });
  const router = useRouter();

  const fetchData = async (token: string) => {
    try {
      console.log('Fetching data with filters:', { searchQuery, selectedLocation, selectedIndustry });
      
      const params = new URLSearchParams();
      
      if (searchQuery.trim()) {
        params.append('search', searchQuery.trim());
      }
      if (selectedLocation) {
        params.append('location', selectedLocation);
      }
      if (selectedIndustry) {
        params.append('industry', selectedIndustry);
      }

      const baseUrl = 'http://localhost:5000/api/investors';
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

      console.log('Fetching from URL:', url);

      const [investorsResponse, statsResponse] = await Promise.all([
        fetch(url, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch('http://localhost:5000/api/investors/stats', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
      ]);

      if (!investorsResponse.ok) {
        throw new Error(`Investors response error: ${investorsResponse.status}`);
      }
      if (!statsResponse.ok) {
        throw new Error(`Stats response error: ${statsResponse.status}`);
      }

      const investorsData = await investorsResponse.json();
      const statsData = await statsResponse.json();

      console.log('Received investors:', investorsData.length);
      console.log('Received stats:', statsData);

      setInvestors(investorsData);
      setStats(statsData);
      setCurrentPage(1);
    } catch (error) {
      console.error('Error in fetchData:', error);
      // Optionally show error to user
      // toast.error('Failed to load investors data');
    } finally {
      setLoading(false);
    }
  };

  const applyAdvancedFilters = async () => {
    setLoading(true);
    try {
      const token = await user.getIdToken();
      console.log('Sending filters:', advancedFilters); // Debug log

      const response = await fetch('http://localhost:5000/api/investors/advanced', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(advancedFilters)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received data:', data); // Debug log
      setInvestors(data);
      setCurrentPage(1); // Reset to first page when filters change
    } catch (error) {
      console.error('Error applying advanced filters:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetFilters = async () => {
    setAdvancedFilters({
      stages: [],
      fund_types: [],
      investment_range: {},
      exits_range: {},
      industries: []
    });
    
    // Fetch original data
    const token = await user.getIdToken();
    fetchData(token);
    setCurrentPage(1);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault(); // Prevent form submission
    if (!user) return;

    setIsSearching(true);
    try {
      const token = await user.getIdToken();
      await fetchData(token);
    } catch (error) {
      console.error('Error during search:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleLocationChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLocation = e.target.value;
    
    // Immediately update UI
    setSelectedLocation(newLocation);
    setLoading(true);
    
    // Clear previous location data
    setInvestors([]); 
    
    if (user) {
      try {
        const token = await user.getIdToken();
        // Create URL with new location
        const params = new URLSearchParams();
        
        if (searchQuery.trim()) {
          params.append('search', searchQuery.trim());
        }
        if (newLocation) { // Use newLocation instead of selectedLocation
          params.append('location', newLocation);
        }
        if (selectedIndustry) {
          params.append('industry', selectedIndustry);
        }

        const baseUrl = 'http://localhost:5000/api/investors';
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

        const response = await fetch(url, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error(`Location filter error: ${response.status}`);
        }

        const data = await response.json();
        setInvestors(data);
        setCurrentPage(1);
      } catch (error) {
        console.error('Error applying location filter:', error);
        // Reset location if there's an error
        setSelectedLocation('');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleIndustryChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newIndustry = e.target.value;
    
    // Immediately update UI
    setSelectedIndustry(newIndustry);
    setLoading(true);
    
    // Clear previous data
    setInvestors([]);
    
    if (user) {
      try {
        const token = await user.getIdToken();
        const params = new URLSearchParams();
        
        if (searchQuery.trim()) {
          params.append('search', searchQuery.trim());
        }
        if (selectedLocation) {
          params.append('location', selectedLocation);
        }
        if (newIndustry) {
          params.append('industry', newIndustry);
        }

        const baseUrl = 'http://localhost:5000/api/investors';
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

        console.log('Fetching with industry filter:', newIndustry);

        const response = await fetch(url, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error(`Industry filter error: ${response.status}`);
        }

        const data = await response.json();
        console.log(`Found ${data.length} investors for industry: ${newIndustry}`);
        setInvestors(data);
        setCurrentPage(1);
      } catch (error) {
        console.error('Error applying industry filter:', error);
        setSelectedIndustry('');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleViewProfile = (investor: any) => {
    if (credits.total === 0) {
      setShowCreditModal(true);
    } else {
      router.push(`/investor/${encodeURIComponent(investor.id)}`);
    }
  };

  const handleGetMoreCredits = () => {
    setShowCreditModal(true);
  };

  const handleLogout = async () => {
    try {
      await auth.signOut();
      router.push('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const fetchCredits = async (userId: string) => {
    try {
      console.log('Fetching credits for user:', userId);
      const userRef = doc(db, 'users', userId);
      const userDoc = await getDoc(userRef);
      
      if (userDoc.exists()) {
        const userData = userDoc.data() as UserData;
        setCredits({
          total: userData.credits || 0
        });
      } else {
        console.log('No user document found');
        setCredits({
          total: 0
        });
      }
    } catch (error) {
      console.error('Error in fetchCredits:', error);
      setCredits({
        total: 0
      });
    }
  };

  const initializeRazorpay = () => {
    return new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = 'https://checkout.razorpay.com/v1/checkout.js';
      
      script.onload = () => {
        resolve(true);
      };
      
      script.onerror = () => {
        resolve(false);
      };

      document.body.appendChild(script);
    });
  };

  const handlePayment = async (plan: PaymentPlan) => {
    setLoadingPayment(true);
    try {
      const res = await initializeRazorpay();

      if (!res) {
        alert('Razorpay SDK failed to load');
        return;
      }

      // Get current user token
      const token = await user.getIdToken();
      
      console.log('Creating order with:', {
        amount: plan.price,
        credits: plan.credits,
        planId: plan.id
      });

      // Make API call to your backend to create order
      const response = await fetch('http://localhost:5000/api/create-order', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          amount: plan.price,
          credits: plan.credits,
          planId: plan.id
        }),
        credentials: 'include',
        mode: 'cors'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Order creation failed:', response.status, errorData);
        throw new Error(`Failed to create order: ${errorData.error || response.statusText}`);
      }

      const order = await response.json();
      
      if (!order.id) {
        throw new Error('Invalid order response: missing order ID');
      }

      console.log('Order created successfully:', order);

      const options = {
        key: RAZORPAY_KEY,
        amount: plan.price * 100, // Convert to paise
        currency: "INR",
        name: "FindMyAngel",
        description: `${plan.name} Plan - ${plan.credits} Credits`,
        order_id: order.id,
        handler: async function (response: any) {
          try {
            const verifyResponse = await fetch('http://localhost:5000/api/verify-payment', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
              },
              body: JSON.stringify({
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_order_id: response.razorpay_order_id,
                razorpay_signature: response.razorpay_signature,
                planId: plan.id,
                credits: plan.credits
              }),
              credentials: 'include',
              mode: 'cors'
            });

            const result = await verifyResponse.json();

            if (result.success) {
              setCredits(prev => ({
                total: prev.total + plan.credits
              }));
              setShowCreditModal(false);
              alert('Payment successful! Credits added to your account.');
            } else {
              throw new Error('Payment verification failed');
            }
          } catch (error) {
            console.error('Payment verification failed:', error);
            alert('Payment verification failed. Please contact support.');
          }
        },
        prefill: {
          name: user?.displayName || '',
          email: user?.email || '',
        },
        theme: {
          color: "#4F46E5",
        },
        modal: {
          ondismiss: function() {
            setLoadingPayment(false);
          }
        }
      };

      const paymentObject = new window.Razorpay(options);
      paymentObject.on('payment.failed', function (response: any) {
        console.error('Payment failed:', response.error);
        alert('Payment failed. Please try again.');
        setLoadingPayment(false);
      });
      paymentObject.open();
      
    } catch (error) {
      console.error('Payment initialization failed:', error);
      alert(`Unable to initialize payment: ${error.message}`);
    } finally {
      setLoadingPayment(false);
    }
  };

  // Add new handler for community join
  const handleJoinCommunity = () => {
    // Open the form modal first
    setShowCommunityModal(true);
  };

  const handleCommunitySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // Get the current user's token
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        throw new Error('Not authenticated');
      }

      console.log('Submitting form data:', communityForm); // Debug log

      // Submit to backend
      const response = await fetch('http://localhost:5000/api/community/join', { // Make sure URL matches your Flask server
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify({
          name: communityForm.name,
          email: communityForm.email,
          startupName: communityForm.startupName,
          location: communityForm.location,
          socialMediaLink: communityForm.socialMediaLink || ''
        })
      });

      console.log('Response status:', response.status); // Debug log

      const data = await response.json();
      console.log('Response data:', data); // Debug log

      if (!response.ok) {
        throw new Error(data.message || 'Failed to join community');
      }

      // Success! Open WhatsApp group and reset form
      window.open('https://chat.whatsapp.com/JBPViBYExVK9UHVreG05Cz', '_blank');
      
      setShowCommunityModal(false);
      setCommunityForm({
        name: '',
        email: '',
        startupName: '',
        location: '',
        socialMediaLink: ''
      });

      // Show success message
      toast.success('Successfully joined the community!');

    } catch (error) {
      console.error('Error joining community:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to join community');
    }
  };

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      if (!user) {
        router.push('/');
      } else {
        console.log('User authenticated:', user.uid); // Debug log
        setUser(user);
        const token = await user.getIdToken();
        fetchData(token);
        fetchCredits(user.uid);
      }
    });

    return () => unsubscribe();
  }, [router]);

  useEffect(() => {
    if (investors.length > 0) {
      setTotalPages(Math.ceil(investors.length / itemsPerPage));
    }
  }, [investors, itemsPerPage]);

  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentInvestors = investors.slice(indexOfFirstItem, indexOfLastItem);

  const paginate = (pageNumber: number) => {
    setCurrentPage(pageNumber);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    const loadInitialData = async () => {
      if (user) {
        try {
          setLoading(true);
          const token = await user.getIdToken();
          await fetchData(token);
        } catch (error) {
          console.error('Error loading initial data:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    loadInitialData();
  }, [user]);

  const redactText = (text: string) => {
    if (!text) return '';
    if (credits.total > 0) return text;
    
    // For names: Keep first letter of each word, replace rest with X
    const words = text.split(' ');
    const redactedWords = words.map(word => {
      if (word.length <= 1) return word;
      return word[0] + 'X'.repeat(word.length - 1);
    });
    return redactedWords.join(' ');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-200'} transition-colors duration-200`}>
      <div className="max-w-[1400px] mx-auto px-6 py-8">
        {/* Header with Stats */}
        <div className="flex justify-between items-center mb-10">
          <div className="flex items-center gap-8">
            {/* Logo */}
            <div className="flex items-center space-x-3 group">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition duration-300"></div>
                <div className="relative bg-gradient-to-r from-indigo-600 to-blue-500 text-white p-2.5 rounded-xl shadow-lg">
                  <svg 
                    className="w-5 h-5" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2.5" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
                  </svg>
                </div>
              </div>
              <div className="flex flex-col">
                <div className="text-2xl tracking-wide" style={{ fontFamily: '"Monomakh", serif' }}>
                  <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 via-indigo-500 to-blue-500">
                    Find
                  </span>
                  <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-blue-400 to-indigo-500">
                    My
                  </span>
                  <span className="bg-clip-text text-transparent bg-gradient-to-br from-indigo-500 via-blue-500 to-indigo-600">
                    Angel
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-[1px] w-4 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
                  <span className="text-[9px] font-medium tracking-[0.2em] text-indigo-600 dark:text-indigo-400" style={{ fontFamily: "'Inter', sans-serif" }}>
                    INVESTOR NETWORK
                  </span>
                  <div className="h-[1px] w-4 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
                </div>
              </div>
            </div>

            <div className="flex gap-8">
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
                <span className="font-medium text-indigo-600">{stats?.total_investors || 0}</span> 
                <span>Investors</span>
              </div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
                <span className="font-medium text-indigo-600">{stats?.total_industries || 0}</span> 
                <span>Industries</span>
              </div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
                <span className="font-medium text-indigo-600">{stats?.total_locations || 0}</span> 
                <span>Locations</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Only show Join Community button if credits > 0 */}
            {credits.total > 0 && (
              <button
                onClick={handleJoinCommunity}
                className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-6 py-2 rounded-lg 
                  transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-[1.02] flex items-center gap-2"
              >
                <svg 
                  className="w-5 h-5" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth="2" 
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                  />
                </svg>
                Join Community
              </button>
            )}
            {/* Show "Get Credits" button if credits === 0 */}
            {credits.total === 0 && (
              <button
                onClick={handleGetMoreCredits}
                className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-6 py-2 rounded-lg 
                  transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-[1.02] flex items-center gap-2"
              >
                <svg 
                  className="w-5 h-5" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth="2" 
                    d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                  />
                </svg>
                Get Credits and Join Community
              </button>
            )}

            <button
              onClick={toggleDarkMode}
              className={`p-2.5 rounded-lg border ${
                darkMode 
                  ? 'bg-gray-800 text-gray-200 border-gray-700' 
                  : 'bg-gray-100 text-gray-600 border-gray-300 shadow-sm hover:bg-gray-200'
              } transition-all duration-200`}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
              darkMode
                ? 'bg-gray-800 border-gray-700'
                : 'bg-gray-100 border-gray-300'
            } shadow-sm`}>
              <svg 
                className={`w-4 h-4 ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className={`font-medium ${
                darkMode ? 'text-indigo-400' : 'text-indigo-600'
              }`}>
                {credits.total} Credits
              </span>
            </div>

            <button
              onClick={handleLogout}
              className={`px-6 py-2 rounded-lg border shadow-sm transition-all duration-200 ${
                darkMode 
                  ? 'border-gray-700 text-gray-300 hover:bg-gray-800' 
                  : 'border-gray-300 bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Logout
            </button>
          </div>
        </div>

        {/* Notice section */}
        <div className={`mb-6 px-4 py-3 rounded-lg shadow-sm border ${
          darkMode 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-gray-100 border-gray-300'
        }`}>
          <div className="flex items-center gap-2">
            <svg 
              className={`w-4 h-4 ${darkMode ? 'text-blue-400' : 'text-blue-500'}`}
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth="2" 
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Viewing an investor profile will consume 1 credit from your account
            </span>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex gap-4 mb-8">
          <form onSubmit={handleSearch} className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search investors..."
              className={`w-full pl-11 pr-4 py-3 border rounded-lg shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 ${
                darkMode 
                  ? 'border-gray-700 bg-gray-800 text-white placeholder-gray-500' 
                  : 'border-gray-300 bg-gray-100 text-gray-900 placeholder-gray-500'
              }`}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </form>

          {/* Location and Industry dropdowns */}
          <select
            className={`px-4 py-3 border rounded-lg shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 min-w-[180px] ${
              darkMode 
                ? 'border-gray-700 bg-gray-800 text-white' 
                : 'border-gray-300 bg-gray-100 text-gray-900'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            value={selectedLocation}
            onChange={handleLocationChange}
            disabled={loading}
          >
            <option value="">All Locations</option>
            {stats?.locations.map((location) => (
              <option key={location} value={location}>{location}</option>
            ))}
          </select>

          <select
            className={`px-4 py-3 border rounded-lg shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 min-w-[200px] ${
              darkMode 
                ? 'border-gray-700 bg-gray-800 text-white' 
                : 'border-gray-300 bg-gray-100 text-gray-900'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            value={selectedIndustry}
            onChange={handleIndustryChange}
            disabled={loading}
          >
            <option value="">All Industries</option>
            {stats?.industries.map((industry) => (
              <option key={industry} value={industry}>{industry}</option>
            ))}
          </select>
        </div>

        {/* Advanced Filters Button */}
        <button
          onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg mb-4 shadow-sm border ${
            darkMode 
              ? 'bg-gray-800 text-white hover:bg-gray-700 border-gray-700' 
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border-gray-300'
          } transition-all duration-200`}
        >
          <Filter size={18} />
          Advanced Filters
          {Object.values(advancedFilters).some(val => 
            Array.isArray(val) ? val.length > 0 : Object.keys(val).length > 0
          ) && (
            <span className="ml-2 px-2 py-0.5 text-xs bg-blue-500 text-white rounded-full">
              Active
            </span>
          )}
        </button>

        {/* Advanced Filters Panel */}
        {showAdvancedFilters && (
          <div className={`mb-6 p-6 rounded-xl shadow-lg ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-100 border-gray-300'
          } border transition-all duration-200`}>
            <div className="flex justify-between items-center mb-6">
              <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Advanced Filters
              </h3>
              <button
                onClick={() => setShowAdvancedFilters(false)}
                className={`p-2 rounded-lg hover:bg-opacity-80 transition-colors ${
                  darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                }`}
              >
                <X size={20} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Investment Stages */}
              <div className="space-y-4">
                <label className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Investment Stages
                </label>
                <div className="flex flex-wrap gap-2">
                  {INVESTMENT_STAGES.map((stage) => {
                    const isSelected = advancedFilters.stages.includes(stage.name);
                    return (
                      <button
                        key={stage.id}
                        onClick={() => {
                          setAdvancedFilters(prev => ({
                            ...prev,
                            stages: prev.stages.includes(stage.name)
                              ? prev.stages.filter(s => s !== stage.name)
                              : [...prev.stages, stage.name]
                          }));
                        }}
                        className={`
                          px-4 py-2 rounded-full text-sm font-medium 
                          transition-all duration-200 border-2
                          ${isSelected 
                            ? `bg-${stage.color}-50 text-${stage.color}-700 border-${stage.color}-500 shadow-sm` 
                            : darkMode
                              ? 'bg-gray-700 text-gray-300 border-gray-600 hover:bg-gray-600'
                              : 'bg-gray-100 text-gray-600 border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                          }
                          focus:outline-none focus:ring-2 focus:ring-offset-2 
                          focus:ring-${stage.color}-500
                        `}
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full bg-${stage.color}-500`}></span>
                          {stage.name}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Fund Types */}
              <div className="space-y-4">
                <label className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Investor Type
                </label>
                <div className="flex flex-wrap gap-2">
                  {INVESTOR_TYPES.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => {
                        setAdvancedFilters(prev => ({
                          ...prev,
                          fund_types: prev.fund_types.includes(type.value)
                            ? prev.fund_types.filter(t => t !== type.value)
                            : [...prev.fund_types, type.value]
                        }));
                      }}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                        advancedFilters.fund_types.includes(type.value)
                          ? 'bg-blue-500 text-white hover:bg-blue-600'
                          : darkMode
                            ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {type.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Investment Range */}
              <div className="space-y-3">
                <label className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Number of Investments
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="relative">
                    <input
                      type="number"
                      min="0"
                      placeholder="Min"
                      value={advancedFilters.investment_range.min || ''}
                      onChange={(e) => {
                        const value = e.target.value ? parseInt(e.target.value) : undefined;
                        setAdvancedFilters(prev => ({
                          ...prev,
                          investment_range: {
                            ...prev.investment_range,
                            min: value
                          }
                        }));
                      }}
                      className={`w-full px-4 py-2.5 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                          : 'bg-gray-100 border-gray-300 text-gray-900 placeholder-gray-400'
                      }`}
                    />
                  </div>
                  <div className="relative">
                    <input
                      type="number"
                      min="0"
                      placeholder="Max"
                      value={advancedFilters.investment_range.max || ''}
                      onChange={(e) => {
                        const value = e.target.value ? parseInt(e.target.value) : undefined;
                        setAdvancedFilters(prev => ({
                          ...prev,
                          investment_range: {
                            ...prev.investment_range,
                            max: value
                          }
                        }));
                      }}
                      className={`w-full px-4 py-2.5 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                          : 'bg-gray-100 border-gray-300 text-gray-900 placeholder-gray-400'
                      }`}
                    />
                  </div>
                </div>
              </div>

              {/* Number of Exits */}
              <div className="space-y-3">
                <label className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Number of Exits
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="relative">
                    <input
                      type="number"
                      min="0"
                      placeholder="Min"
                      value={advancedFilters.exits_range.min || ''}
                      onChange={(e) => {
                        const value = e.target.value ? parseInt(e.target.value) : undefined;
                        setAdvancedFilters(prev => ({
                          ...prev,
                          exits_range: {
                            ...prev.exits_range,
                            min: value
                          }
                        }));
                      }}
                      className={`w-full px-4 py-2.5 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                          : 'bg-gray-100 border-gray-300 text-gray-900 placeholder-gray-400'
                      }`}
                    />
                  </div>
                  <div className="relative">
                    <input
                      type="number"
                      min="0"
                      placeholder="Max"
                      value={advancedFilters.exits_range.max || ''}
                      onChange={(e) => {
                        const value = e.target.value ? parseInt(e.target.value) : undefined;
                        setAdvancedFilters(prev => ({
                          ...prev,
                          exits_range: {
                            ...prev.exits_range,
                            max: value
                          }
                        }));
                      }}
                      className={`w-full px-4 py-2.5 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                          : 'bg-gray-100 border-gray-300 text-gray-900 placeholder-gray-400'
                      }`}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-8 flex justify-end gap-4">
              <button
                onClick={resetFilters}
                className={`px-6 py-2.5 rounded-lg transition-colors ${
                  darkMode 
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Reset
              </button>
              <button
                onClick={applyAdvancedFilters}
                className="px-6 py-2.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                Apply Filters
              </button>
            </div>
          </div>
        )}

        {/* Table Container */}
        <div className={`rounded-lg border shadow-sm ${
          darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-100 border-gray-300'
        }`}>
          {/* Enhanced Bright Gradient Disclaimer with Larger Font */}
          {credits.total === 0 && (
            <div className="relative overflow-hidden rounded-t-lg">
              <div className="absolute inset-0 bg-gradient-to-r from-violet-500/40 via-blue-500/40 to-fuchsia-500/40 animate-gradient-slow"></div>
              <div className={`px-8 py-5 bg-gradient-to-r ${
                darkMode 
                  ? 'from-violet-800/95 via-blue-900/90 to-fuchsia-800/95' 
                  : 'from-violet-50/95 via-blue-100/95 to-fuchsia-50/95'
              } backdrop-blur-md border-b border-blue-400/50`}>
                <div className="flex justify-center items-center">
                  <p className={`text-[17px] font-semibold bg-gradient-to-r ${
                    darkMode 
                      ? 'from-violet-300 via-blue-200 to-fuchsia-300' 
                      : 'from-violet-600 via-blue-600 to-fuchsia-600'
                  } bg-clip-text text-transparent tracking-wide`}>
                    Purchase credits to unlock complete investor profiles and contact information
                  </p>
                </div>
              </div>
            </div>
          )}

          <table className="min-w-full divide-y divide-gray-200">
            <thead>
              <tr className={darkMode ? 'bg-gray-700' : 'bg-gray-200/75'}>
                <th scope="col" className={`px-6 py-4 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Investor Name
                </th>
                <th scope="col" className={`px-6 py-4 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Headline
                </th>
                <th scope="col" className={`px-6 py-4 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Company
                </th>
                <th scope="col" className={`px-6 py-4 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Location
                </th>
                <th scope="col" className={`px-6 py-4 text-right text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                  Action
                </th>
              </tr>
            </thead>
            <tbody className={`divide-y ${
              darkMode ? 'bg-gray-800 divide-gray-700' : 'bg-gray-100 divide-gray-300'
            }`}>
              {currentInvestors.map((investor) => (
                <tr key={investor.id} className={`${
                  darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                } transition-colors duration-150`}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className={`h-10 w-10 rounded-full ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} flex items-center justify-center ${darkMode ? 'text-gray-300' : 'text-gray-600'} font-medium`}>
                        {investor.name.charAt(0)}
                      </div>
                      <div className="ml-4">
                        <div className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {redactText(investor.name)}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className={`px-6 py-4 ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                    <div className="flex flex-col gap-2">
                      <div className="text-sm">{redactText(investor.headline)}</div>
                      <div className="flex flex-wrap gap-1.5 mt-1">
                        {investor.fund_focus.slice(0, 10).map((focus, index) => (
                          <span
                            key={index}
                            className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                              darkMode 
                                ? 'bg-gray-700 text-gray-300' 
                                : 'bg-gray-100 text-gray-600'
                            }`}
                          >
                            {focus}
                          </span>
                        ))}
                        {investor.fund_focus.length > 10 && (
                          <span
                            className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                              darkMode 
                                ? 'bg-gray-600 text-gray-300' 
                                : 'bg-gray-200 text-gray-600'
                            }`}
                          >
                            +{investor.fund_focus.length - 10}
                          </span>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                    {redactText(investor.company_name)}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                    {investor.location.join(', ')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <button 
                      onClick={() => handleViewProfile(investor)}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-500/10 font-medium text-sm transition-all duration-200"
                    >
                      <Eye size={16} />
                      View Profile
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div className={`px-6 py-4 border-t ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-100 border-gray-300'
          }`}>
            <div className="flex items-center justify-between">
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                Showing {indexOfFirstItem + 1} to {Math.min(indexOfLastItem, investors.length)} of {investors.length} results
              </div>
              
              <div className="flex items-center gap-2">
                <button
                  onClick={() => paginate(currentPage - 1)}
                  disabled={currentPage === 1}
                  className={`p-2 rounded-lg ${
                    darkMode 
                      ? 'bg-gray-700 text-gray-300 disabled:bg-gray-900 disabled:text-gray-600' 
                      : 'bg-gray-100 text-gray-600 disabled:bg-gray-50 disabled:text-gray-400'
                  } hover:bg-blue-500 hover:text-white disabled:hover:bg-gray-100 disabled:hover:text-gray-400 transition-colors`}
                >
                  <ChevronLeft size={20} />
                </button>
                
                {/* Page numbers */}
                <div className="flex items-center gap-1">
                  {[...Array(totalPages)].map((_, index) => {
                    const pageNumber = index + 1;
                    // Show first page, last page, current page, and pages around current page
                    if (
                      pageNumber === 1 ||
                      pageNumber === totalPages ||
                      (pageNumber >= currentPage - 1 && pageNumber <= currentPage + 1)
                    ) {
                      return (
                        <button
                          key={pageNumber}
                          onClick={() => paginate(pageNumber)}
                          className={`px-4 py-2 rounded-lg transition-colors ${
                            currentPage === pageNumber
                              ? 'bg-blue-500 text-white'
                              : darkMode
                              ? 'bg-gray-700 text-gray-300 hover:bg-blue-500 hover:text-white'
                              : 'bg-gray-100 text-gray-600 hover:bg-blue-500 hover:text-white'
                          }`}
                        >
                          {pageNumber}
                        </button>
                      );
                    } else if (
                      pageNumber === currentPage - 2 ||
                      pageNumber === currentPage + 2
                    ) {
                      return (
                        <span
                          key={pageNumber}
                          className={`px-4 py-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
                        >
                          ...
                        </span>
                      );
                    }
                    return null;
                  })}
                </div>

                <button
                  onClick={() => paginate(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className={`p-2 rounded-lg ${
                    darkMode 
                      ? 'bg-gray-700 text-gray-300 disabled:bg-gray-900 disabled:text-gray-600' 
                      : 'bg-gray-100 text-gray-600 disabled:bg-gray-50 disabled:text-gray-400'
                  } hover:bg-blue-500 hover:text-white disabled:hover:bg-gray-100 disabled:hover:text-gray-400 transition-colors`}
                >
                  <ChevronRight size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Credit Purchase Modal */}
      {showCreditModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className={`${
            darkMode ? 'bg-gray-800' : 'bg-white'
          } rounded-2xl shadow-xl max-w-5xl w-full transform transition-all duration-300 scale-100`}>
            {/* Modal Header with Enhanced Gradient */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/20 via-blue-500/20 to-purple-500/20 backdrop-blur-sm"></div>
              <div className="p-8 relative">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className={`text-3xl font-bold bg-gradient-to-r from-indigo-500 via-blue-500 to-purple-500 bg-clip-text text-transparent animate-gradient`}>
                      Choose Your Plan
                    </h2>
                    <p className={`mt-2 text-lg bg-gradient-to-r from-indigo-400 via-blue-400 to-purple-400 bg-clip-text text-transparent`}>
                      Select the perfect plan to unlock investor profiles and grow your startup
                    </p>
                  </div>
                  <button 
                    onClick={() => setShowCreditModal(false)}
                    className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                  >
                    <X size={24} className="text-gray-400 hover:text-white transition-colors" />
                  </button>
                </div>
              </div>
            </div>

            {/* Pricing Cards Grid */}
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Starter Plan */}
                <div className={`relative p-6 rounded-xl shadow-lg border ${
                  darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'
                } transform hover:scale-105 transition-all duration-300`}>
                  <div className={`text-xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Starter
                  </div>
                  <div className="flex items-baseline gap-2 mb-2">
                    <div className="text-3xl font-bold text-indigo-600">₹499</div>
                    <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
                  </div>
                  <div className={`mb-6 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <span className="text-xl font-semibold text-indigo-600">50</span>
                    <span> tokens</span>
                  </div>
                  <ul className={`space-y-3 mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      View 50 investor profiles
                    </li>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      Investment focus areas
                    </li>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      WhatsApp community access
                    </li>
                  </ul>
                  <button 
                    onClick={() => handlePayment({ id: 'starter', price: 499, credits: 50 })}
                    className={`w-full py-2.5 rounded-lg font-medium transition-colors ${
                      darkMode 
                        ? 'bg-gray-700 text-white hover:bg-gray-600' 
                        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                    }`}
                  >
                    Get Started
                  </button>
                </div>

                {/* Pro Plan */}
                <div className={`relative p-6 rounded-xl shadow-xl border-2 border-indigo-500 ${
                  darkMode ? 'bg-gray-800' : 'bg-white'
                } transform hover:scale-105 transition-all duration-300`}>
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-gradient-to-r from-indigo-600 to-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium shadow-lg">
                      Most Popular
                    </span>
                  </div>
                  <div className={`text-xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Pro
                  </div>
                  <div className="flex items-baseline gap-2 mb-2">
                    <div className="text-3xl font-bold text-indigo-600">₹1,699</div>
                    <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
                  </div>
                  <div className={`mb-6 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <span className="text-xl font-semibold text-indigo-600">200</span>
                    <span> tokens</span>
                  </div>
                  <ul className={`space-y-3 mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      View 200 investor profiles
                    </li>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      Investment focus areas
                    </li>
                    <li className="flex items-center">
                      <Check className="w-5 h-5 text-indigo-500 mr-2" />
                      WhatsApp community access
                    </li>
                  </ul>
                  <button 
                    onClick={() => handlePayment({ id: 'pro', price: 1699, credits: 200 })}
                    className="w-full py-2.5 rounded-lg font-medium bg-gradient-to-r from-indigo-600 to-blue-500 text-white hover:from-indigo-700 hover:to-blue-600 transition-all duration-200"
                  >
                    Get Started
                  </button>
                </div>

                {/* Enterprise Plan */}
                <div className={`relative p-6 rounded-xl ${
                  darkMode 
                    ? 'bg-gradient-to-b from-gray-800 via-gray-800/80 to-purple-900/10' 
                    : 'bg-gradient-to-b from-white via-gray-50 to-purple-50'
                } border border-indigo-500/20 transform hover:scale-105 transition-all duration-300 hover:shadow-xl group`}>
                  <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-indigo-500/10 rounded-t-xl opacity-75 group-hover:opacity-100 transition-opacity"></div>
                  <div className="relative">
                    <div className={`inline-flex items-center space-x-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      <span className="text-xl font-bold">Enterprise</span>
                      <div className="h-5 w-px bg-gradient-to-b from-purple-500 to-transparent"></div>
                    </div>
                    <div className="flex items-baseline gap-2 mt-4">
                      <div className="text-4xl font-bold bg-gradient-to-r from-purple-600 via-blue-500 to-indigo-500 bg-clip-text text-transparent">₹3,999</div>
                      <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
                    </div>
                    <div className={`mt-2 mb-6`}>
                      <span className="text-2xl font-semibold bg-gradient-to-r from-purple-600 via-blue-500 to-indigo-500 bg-clip-text text-transparent">500</span>
                      <span className={`${darkMode ? 'text-gray-400' : 'text-gray-500'}`}> tokens</span>
                    </div>
                    <ul className={`space-y-3 mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      <li className="flex items-center">
                        <div className="p-1 rounded-full bg-gradient-to-r from-purple-500/20 to-blue-500/20 mr-2">
                          <Check className="w-4 h-4 text-purple-500" />
                        </div>
                        View 500 investor profiles
                      </li>
                      <li className="flex items-center">
                        <div className="p-1 rounded-full bg-gradient-to-r from-purple-500/20 to-blue-500/20 mr-2">
                          <Check className="w-4 h-4 text-purple-500" />
                        </div>
                        Investment focus areas
                      </li>
                      <li className="flex items-center">
                        <div className="p-1 rounded-full bg-gradient-to-r from-purple-500/20 to-blue-500/20 mr-2">
                          <Check className="w-4 h-4 text-purple-500" />
                        </div>
                        WhatsApp community access
                      </li>
                      <li className="flex items-center">
                        <div className="p-1 rounded-full bg-gradient-to-r from-purple-500/20 to-blue-500/20 mr-2">
                          <Check className="w-4 h-4 text-purple-500" />
                        </div>
                        Priority support 24/7
                      </li>
                      {/* <li className="flex items-center">
                        <div className="p-1 rounded-full bg-gradient-to-r from-purple-500/20 to-blue-500/20 mr-2">
                          <Check className="w-4 h-4 text-purple-500" />
                        </div>
                        Custom export options
                      </li> */}
                    </ul>
                    <button 
                      onClick={() => handlePayment({ id: 'enterprise', price: 3999, credits: 500 })}
                      className={`w-full py-3 rounded-lg font-medium bg-gradient-to-r hover:shadow-lg transition-all duration-200 ${
                        darkMode 
                          ? 'from-purple-900 to-indigo-900 hover:from-purple-800 hover:to-indigo-800 text-white border border-purple-700/50' 
                          : 'from-purple-50 to-indigo-50 hover:from-purple-100 hover:to-indigo-100 text-gray-900 border border-purple-200'
                      }`}
                    >
                      Get Started
                    </button>
                  </div>
                </div>
              </div>

              {/* Enhanced Footer Note with more vibrant gradient */}
              <div className={`text-center mt-8 p-4 rounded-xl ${
                darkMode 
                  ? 'bg-gradient-to-r from-gray-800/50 via-indigo-900/30 to-purple-900/30' 
                  : 'bg-gradient-to-r from-gray-50 via-indigo-50/50 to-purple-50/50'
              } border border-indigo-500/10`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  All prices are in Indian Rupees (INR). Tokens never expire and can be used anytime.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Add Community Modal */}
      {showCommunityModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className={`${
            darkMode ? 'bg-gray-800' : 'bg-white'
          } rounded-2xl shadow-xl w-[480px] transform transition-all duration-300 scale-100`}>
            {/* Modal Header with Blue-Purple Gradient */}
            <div className="relative overflow-hidden rounded-t-2xl">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600"></div>
              <div className="relative p-5">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-xl font-semibold text-white">
                      Join Our Investor Community
                    </h2>
                    <p className="mt-1 text-sm text-white/90">
                      Connect with fellow founders and find your perfect investor match
                    </p>
                  </div>
                  <button 
                    onClick={() => setShowCommunityModal(false)}
                    className="p-1.5 rounded-lg transition-colors hover:bg-white/10"
                  >
                    <X size={18} className="text-white" />
                  </button>
                </div>
              </div>
            </div>

            {/* Form Section */}
            <form onSubmit={handleCommunitySubmit} className="p-5 space-y-4">
              <div className="space-y-4">
                {/* Name Input */}
                <div className="group">
                  <label className={`block mb-1.5 text-sm font-medium transition-colors ${
                    darkMode ? 'text-gray-300 group-focus-within:text-indigo-400' : 'text-gray-700 group-focus-within:text-indigo-600'
                  }`}>
                    Your Name *
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      required
                      value={communityForm.name}
                      onChange={(e) => setCommunityForm({...communityForm, name: e.target.value})}
                      className={`w-full px-3.5 py-2 rounded-lg border transition-all duration-200 focus:ring-2 focus:ring-offset-2 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white focus:border-indigo-500 focus:ring-indigo-500/20' 
                          : 'bg-white border-gray-300 text-gray-900 focus:border-indigo-500 focus:ring-indigo-500/20'
                      }`}
                      placeholder="John Doe"
                    />
                  </div>
                </div>

                {/* Email Input */}
                <div className="group">
                  <label className={`block mb-1.5 text-sm font-medium transition-colors ${
                    darkMode ? 'text-gray-300 group-focus-within:text-indigo-400' : 'text-gray-700 group-focus-within:text-indigo-600'
                  }`}>
                    Email *
                  </label>
                  <div className="relative">
                    <input
                      type="email"
                      required
                      value={communityForm.email}
                      onChange={(e) => setCommunityForm({...communityForm, email: e.target.value})}
                      className={`w-full px-3.5 py-2 rounded-lg border transition-all duration-200 focus:ring-2 focus:ring-offset-2 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white focus:border-indigo-500 focus:ring-indigo-500/20' 
                          : 'bg-white border-gray-300 text-gray-900 focus:border-indigo-500 focus:ring-indigo-500/20'
                      }`}
                      placeholder="john@example.com"
                    />
                  </div>
                </div>

                {/* Startup Name Input */}
                <div className="group">
                  <label className={`block mb-1.5 text-sm font-medium transition-colors ${
                    darkMode ? 'text-gray-300 group-focus-within:text-indigo-400' : 'text-gray-700 group-focus-within:text-indigo-600'
                  }`}>
                    Startup Name *
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      required
                      value={communityForm.startupName}
                      onChange={(e) => setCommunityForm({...communityForm, startupName: e.target.value})}
                      className={`w-full px-3.5 py-2 rounded-lg border transition-all duration-200 focus:ring-2 focus:ring-offset-2 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white focus:border-indigo-500 focus:ring-indigo-500/20' 
                          : 'bg-white border-gray-300 text-gray-900 focus:border-indigo-500 focus:ring-indigo-500/20'
                      }`}
                      placeholder="Amazing Startup Inc."
                    />
                  </div>
                </div>

                {/* Location Input */}
                <div className="group">
                  <label className={`block mb-1.5 text-sm font-medium transition-colors ${
                    darkMode ? 'text-gray-300 group-focus-within:text-indigo-400' : 'text-gray-700 group-focus-within:text-indigo-600'
                  }`}>
                    Startup Location *
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      required
                      value={communityForm.location}
                      onChange={(e) => setCommunityForm({...communityForm, location: e.target.value})}
                      className={`w-full px-3.5 py-2 rounded-lg border transition-all duration-200 focus:ring-2 focus:ring-offset-2 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white focus:border-indigo-500 focus:ring-indigo-500/20' 
                          : 'bg-white border-gray-300 text-gray-900 focus:border-indigo-500 focus:ring-indigo-500/20'
                      }`}
                      placeholder="Mumbai, India"
                    />
                  </div>
                </div>

                {/* Social Media Link Input */}
                <div className="group">
                  <label className={`block mb-1.5 text-sm font-medium transition-colors ${
                    darkMode ? 'text-gray-300 group-focus-within:text-indigo-400' : 'text-gray-700 group-focus-within:text-indigo-600'
                  }`}>
                    Social Media Link (Optional)
                  </label>
                  <div className="relative">
                    <input
                      type="url"
                      value={communityForm.socialMediaLink}
                      onChange={(e) => setCommunityForm({...communityForm, socialMediaLink: e.target.value})}
                      className={`w-full px-3.5 py-2 rounded-lg border transition-all duration-200 focus:ring-2 focus:ring-offset-2 ${
                        darkMode 
                          ? 'bg-gray-700 border-gray-600 text-white focus:border-indigo-500 focus:ring-indigo-500/20' 
                          : 'bg-white border-gray-300 text-gray-900 focus:border-indigo-500 focus:ring-indigo-500/20'
                      }`}
                      placeholder="https://linkedin.com/company/your-startup"
                    />
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                className="w-full py-2.5 px-4 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white rounded-lg font-medium 
                  transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2
                  disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Join WhatsApp Community
              </button>

              {/* Footer Text */}
              <p className={`text-center text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                By joining, you'll be connected with fellow founders and investors
              </p>
            </form>
          </div>
        </div>
      )}
    </div>
  );
} 