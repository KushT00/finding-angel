/* eslint-disable react-hooks/rules-of-hooks */
/* eslint-disable @next/next/no-img-element */
"use client";
import { useState, useEffect } from "react";
import { GoogleAuthProvider, signInWithPopup, signOut  } from "firebase/auth";
import { auth } from "../lib/firebase";
import { doc, getDoc, setDoc, serverTimestamp } from "firebase/firestore";
import { db } from "../lib/firebase";
import { useRouter } from 'next/navigation';
import Head from 'next/head';


const provider = new GoogleAuthProvider();

export default function Home() {
  
  // Define User Type
  type User = {
    name: string;
    picture: string;
  };
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Check system preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setDarkMode(true);
      document.documentElement.classList.add('dark');
    }

    // Listen for auth state changes
    const unsubscribe = auth.onAuthStateChanged(async (firebaseUser) => {
      if (firebaseUser) {
        const token = await firebaseUser.getIdToken();
        // Verify token with backend
        try {
          const response = await fetch('https://findmyangelapi.vercel.app/auth/verify', {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            credentials: 'include',
          });

          if (response.ok) {
            const userData = await response.json();
            setUser(userData);
          }
        } catch (error) {
          console.error('Error verifying token:', error);
        }
      } else {
        setUser(null);
      }
    });

    return () => unsubscribe();
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    // Add or remove dark mode class from body
    document.documentElement.classList.toggle('dark');
  };

  const handleSignIn = async () => {
    try {
      setIsLoading(true);
      const result = await signInWithPopup(auth, provider);

      // Add/Update user in Firestore
      const userRef = doc(db, 'users', result.user.uid);
      const userDoc = await getDoc(userRef);

      if (!userDoc.exists()) {
        // Create new user document if it doesn't exist
        await setDoc(userRef, {
          username: result.user.displayName,
          email: result.user.email,
          uuid: result.user.uid,
          credits: 0,
          createdAt: serverTimestamp()
        });
      }

      // Redirect to dashboard after successful sign in
      router.push('/dashboard');

    } catch (error) {
      console.error('Error signing in:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      await fetch('https://findmyangelapi.vercel.app/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'dark bg-gray-900' : 'bg-gradient-to-b from-gray-50 to-white'}`}>
      {/* Navbar */}
      <Head>
        <title>FindMyAngel - Connect with Investors</title>
        <meta name="description" content="Find and connect with top angel investors and VCs worldwide." />
      </Head>
      <nav className={`fixed top-0 left-0 right-0 z-50 ${darkMode ? 'bg-gray-900/80' : 'bg-white/80'} backdrop-blur-sm border-b ${darkMode ? 'border-gray-800' : 'border-gray-100'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3 group">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition duration-300"></div>
                <div className="relative bg-gradient-to-r from-indigo-600 to-blue-500 text-white p-2.5 rounded-xl shadow-lg">
                  <svg
                    className="w-6 h-6"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />
                  </svg>
                </div>
              </div>
              <div className="flex flex-col">
                <div className="text-4xl tracking-wide" style={{ fontFamily: '"Monomakh", serif' }}>
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
                  <div className="h-[1px] w-5 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
                  <span className="text-[10px] font-medium tracking-[0.2em] text-indigo-600 dark:text-indigo-400" style={{ fontFamily: "'Inter', sans-serif" }}>
                    INVESTOR NETWORK
                  </span>
                  <div className="h-[1px] w-5 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
                </div>
              </div>
            </div>

            {/* Right side */}
            <div className="flex items-center space-x-4">
              <button
                onClick={toggleDarkMode}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-100'} transition-colors duration-200`}
              >
                {darkMode ? (
                  <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                )}
              </button>
              {!user ? (
                <button
                  onClick={handleSignIn}
                  className="flex items-center gap-2 px-4 py-2 text-white bg-blue-600 hover:bg-blue-700 
                  transition-colors duration-200 rounded-lg font-medium shadow-sm hover:shadow-md"
                >
                  {isLoading ? (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    "Sign In"
                  )}
                </button>
              ) : (
                <div className="flex items-center space-x-4">
                  <img
                    src={user?.picture || "/default-avatar.png"}
                    alt={user?.name || "User"}
                    className="w-8 h-8 rounded-full"
                  />
                  <button
                    onClick={handleSignOut}
                    className="px-4 py-2 text-gray-700 hover:text-gray-900 bg-gray-100 
                    hover:bg-gray-200 rounded-lg font-medium transition-colors duration-200"
                  >
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className={`pt-32 pb-20 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="mb-6" style={{ fontFamily: '"Monomakh", serif' }}>
              <div className={`text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Find <span className="text-indigo-600">2200+ investors</span> ready to fund your startup idea
              </div>
            </h1>

            <p className={`text-xl md:text-2xl mb-12 leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} style={{ fontFamily: "'Inter', sans-serif" }}>
              Search and connect with top angel investors and VCs worldwide
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-4 rounded-xl font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 transform hover:scale-105 transition-all duration-200 shadow-lg" onClick={handleSignIn}>
                Get Started
              </button>
              <button
               onClick={() => router.push("/about")}
              className={`px-8 py-4 rounded-xl font-medium ${darkMode ? 'bg-gray-800 text-white hover:bg-gray-700' : 'bg-white text-gray-900 hover:bg-gray-100'} border border-gray-200 transform hover:scale-105 transition-all duration-200 shadow-lg`}>
                Learn More
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 mt-16 max-w-3xl mx-auto">
            <div className={`border rounded-xl p-6 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-transform duration-300`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">2200+</div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mt-2 font-medium`}>Verified Investors</div>
            </div>
            <div className={`border rounded-xl p-6 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-transform duration-300`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">100+</div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mt-2 font-medium`}>Investment Sectors</div>
            </div>
            <div className={`border rounded-xl p-6 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-transform duration-300`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">100%</div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mt-2 font-medium`}>Verified Profiles</div>
            </div>
          </div>
        </div>
      </div>

      {/* Fundraising Comparison Section */}
      <div className={`py-20 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="container mx-auto px-6">
          <h2 className={`text-3xl font-bold text-center mb-16 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>
            Traditional fundraising <span className="text-indigo-600">v/s</span> Modern fundraising
          </h2>

          <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
            {/* Traditional Fundraising */}
            <div className={`p-8 rounded-2xl border ${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-100'}`}>
              <div className={`text-2xl font-bold mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} style={{ fontFamily: '"Monomakh", serif' }}>
                Traditional
              </div>
              <ul className="space-y-6">
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-red-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Getting frustrated looking for contacts
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-red-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Wasting time researching about investors
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-red-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Constantly sending 1000s of LinkedIn requests
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-red-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Giving 2% commission to platforms
                  </span>
                </li>
              </ul>
            </div>

            {/* Modern Fundraising */}
            <div className={`p-8 rounded-2xl border-2 border-indigo-500 ${darkMode ? 'bg-gray-800/50' : 'bg-white'}`}>
              <div className="text-2xl font-bold mb-8 text-indigo-600" style={{ fontFamily: '"Monomakh", serif' }}>
                Modern
              </div>
              <ul className="space-y-6">
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-green-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Get 1000+ verified investors at your fingertips
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-green-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Simply email them your pitch deck
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-green-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Save 100s of hours and focus on your startup
                  </span>
                </li>
                <li className="flex items-start">
                  <svg className="w-6 h-6 text-green-500 mr-3 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Save money by giving 0% commission
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>


      {/* Pricing Section */}
      <div id="pricing" className={`py-20 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="container px-4 mx-auto">
          <h2 className={`text-3xl md:text-4xl font-bold text-center mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>
            Simple, Token-Based Pricing
          </h2>
          <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} max-w-2xl mx-auto mb-8`}>
            Choose the perfect plan for your startup. All plans include access to our exclusive WhatsApp investor community.
          </p>

          {/* Pricing Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 max-w-6xl mx-auto">
            {/* Free Plan */}
            <div className={`relative p-8 rounded-2xl shadow-lg border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-all duration-300`}>
              <div className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>Free</div>
              <div className="flex items-baseline gap-2 mb-2">
                <div className="text-4xl font-bold text-indigo-600" style={{ fontFamily: '"Monomakh", serif' }}>₹0</div>
                <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>forever</span>
              </div>
              <div className={`mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Start exploring
              </div>
              <ul className={`space-y-4 mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                <li className="flex items-start">
                  <svg className="w-5 h-5 text-green-500 mr-2 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span>Access to full search and filtering of investors</span>
                </li>
                <li className="flex items-start">
                  <svg className="w-5 h-5 text-green-500 mr-2 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span>Read summary of investors</span>
                </li>
                <li className="flex items-start">
                  <svg className="w-5 h-5 text-green-500 mr-2 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  <span>Browse investors from 50+ countries</span>
                </li>
              </ul>
              <div className="flex justify-center">
                <button className={`w-full py-3 rounded-lg font-medium transition-colors duration-200 ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-100 text-gray-900 hover:bg-gray-200'}`} onClick={handleSignIn}>
                  Get Started
                </button>
              </div>
            </div>

            {/* Starter Plan */}
            <div className={`relative p-8 rounded-2xl shadow-lg border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-all duration-300`}>
              <div className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>Starter</div>
              <div className="flex items-baseline gap-2 mb-2">
                <div className="text-4xl font-bold text-indigo-600" style={{ fontFamily: '"Monomakh", serif' }}>₹499</div>
                <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
              </div>
              <div className={`mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <span className="text-2xl font-semibold text-indigo-600">50</span>
                <span> tokens</span>
              </div>
              <ul className={`space-y-4 mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  View 50 investor profiles
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Investment focus areas
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  WhatsApp community access
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Sharable Link
                </li>
              </ul>
              <div className="flex justify-center">
                <button className={`w-full py-3 rounded-lg font-medium transition-colors duration-200 ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-100 text-gray-900 hover:bg-gray-200'}`} onClick={handleSignIn}>
                  Get Started
                </button>
              </div>
            </div>

            {/* Pro Plan with Most Popular tag */}
            <div className={`relative p-8 rounded-2xl shadow-xl border-2 border-indigo-500 ${darkMode ? 'bg-gray-800' : 'bg-white'} transform hover:scale-105 transition-all duration-300`}>
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <span className="bg-gradient-to-r from-indigo-600 to-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium shadow-lg">
                  Most Popular
                </span>
              </div>
              <div className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>Pro</div>
              <div className="flex items-baseline gap-2 mb-2">
                <div className="text-4xl font-bold text-indigo-600" style={{ fontFamily: '"Monomakh", serif' }}>₹1,699</div>
                <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
              </div>
              <div className={`mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <span className="text-2xl font-semibold text-indigo-600">200</span>
                <span> tokens</span>
              </div>
              <ul className={`space-y-4 mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  View 200 investor profiles
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Investment focus areas
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  WhatsApp community access
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Priority support
                </li>
              </ul>
              <button className="w-full py-3 rounded-lg font-medium bg-gradient-to-r from-indigo-600 to-blue-500 text-white hover:from-indigo-700 hover:to-blue-600 transition-all duration-200" onClick={handleSignIn}>
                Get Started
              </button>
            </div>

            {/* Enterprise Plan */}
            <div className={`relative p-8 rounded-2xl shadow-lg border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} transform hover:scale-105 transition-all duration-300`}>
              <div className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>Enterprise</div>
              <div className="flex items-baseline gap-2 mb-2">
                <div className="text-4xl font-bold text-indigo-600" style={{ fontFamily: '"Monomakh", serif' }}>₹3,999</div>
                <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>one-time</span>
              </div>
              <div className={`mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <span className="text-2xl font-semibold text-indigo-600">500</span>
                <span> tokens</span>
              </div>
              <ul className={`space-y-4 mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  View 500 investor profiles
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Investment focus areas
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  WhatsApp community access
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  Priority support
                </li>
              </ul>
              <button className={`w-full py-3 rounded-lg font-medium transition-colors duration-200 ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-100 text-gray-900 hover:bg-gray-200'}`} onClick={handleSignIn}>
                Get Started
              </button>
            </div>
          </div>

          {/* Optional: Add a pricing comparison note */}
          <div className="text-center mt-12 max-w-2xl mx-auto">
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              All prices are in Indian Rupees (INR). The more tokens you buy, the less you pay per token.
              Tokens never expire and can be used anytime.
            </p>
          </div>
        </div>
      </div>


      {/* FAQ Section */}
      <div id="faq" className={`py-20 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="container px-4 mx-auto">
          <h2 className={`text-3xl md:text-4xl font-bold text-center mb-12 md:mb-16 ${darkMode ? 'text-white' : 'text-[#121212]'}`} style={{ fontFamily: '"Monomakh", serif' }}>
            FAQ
          </h2>

          <div className="max-w-2xl mx-auto">
            <div className="space-y-4">
              {(() => {
                const [openIndex, setOpenIndex] = useState(0); // Track which FAQ is open

                const faqs = [
                  {
                    question: "How much do these investors invest?",
                    answer: "Angel investors typically invest ₹5-50 Lakhs (in India) or $5000 to $100,000 (in the US). That's enough to test your market, work on your ideas, and to grow your startup."
                  },
                  {
                    question: "Who is this for?",
                    answer: "This platform is designed for startup founders looking for seed funding or early-stage investment. Whether you're at the idea stage or already have a product, our investors can help you grow."
                  },
                  {
                    question: "How do I contact these investors?",
                    answer: "Once you purchase tokens, you'll get access to investor profiles with their contact information. You can then reach out to them directly via email with your pitch deck."
                  },
                  {
                    question: "Should I raise funding or bootstrap?",
                    answer: "It depends on your business model and growth plans. Funding can help you scale faster, but bootstrapping gives you more control. We recommend discussing your specific case with mentors or advisors."
                  }
                ];

                return faqs.map((faq, index) => (
                  <div key={index} className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                    <button
                      onClick={() => setOpenIndex(openIndex === index ? -1 : index)}
                      className="flex items-center justify-between w-full py-4 text-left transition-colors duration-200 hover:text-indigo-600"
                    >
                      <span className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} hover:text-indigo-600 transition-colors duration-200`}>
                        {faq.question}
                      </span>
                      <svg
                        className={`w-6 h-6 transition-transform duration-200 ${openIndex === index ? 'rotate-180' : ''} ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M19 9l-7 7-7-7"
                        />
                      </svg>
                    </button>
                    <div
                      className={`overflow-hidden transition-all duration-300 ease-in-out ${openIndex === index ? 'max-h-40 opacity-100' : 'max-h-0 opacity-0'
                        }`}
                    >
                      <div className={`pb-4 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                        {faq.answer}
                      </div>
                    </div>
                  </div>
                ));
              })()}
            </div>
          </div>
        </div>
      </div>

      {/* Contact Section */}
      <div className={`py-20 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto">
            <h2 className={`text-3xl font-bold text-center mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`} style={{ fontFamily: '"Monomakh", serif' }}>
              Get in Touch
            </h2>
            <p className={`text-center mb-12 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Have questions? We&apos;d love to hear from you. Send us a message and we&apos;ll respond as soon as possible.
            </p>

            <div className={`p-8 rounded-2xl shadow-lg ${darkMode ? 'bg-gray-800/50 border border-gray-700' : 'bg-white border border-gray-100'}`}>
              <form onSubmit={async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target as HTMLFormElement);

                try {
                  const response = await fetch('https://findmyangelapi.vercel.app/api/contact', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                      name: formData.get('name'),
                      email: formData.get('email'),
                      subject: formData.get('subject'),
                      message: formData.get('message')
                    })
                  });

                  if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Failed to send message');
                  }

                  alert('Message sent successfully!');
                  (e.target as HTMLFormElement).reset();
                } catch (error) {
                  console.error('Error:', error);
                  alert(error instanceof Error ? error.message : 'Failed to send message. Please try again.');
                }
              }} className={`space-y-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Name</span>
                      <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      name="name"
                      required
                      placeholder="Your name"
                      className={`w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${darkMode
                        ? 'bg-gray-700/50 border-gray-600 text-white placeholder-gray-400'
                        : 'bg-gray-50 border-gray-200 placeholder-gray-400'
                        }`}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Email</span>
                      <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="email"
                      name="email"
                      required
                      placeholder="you@example.com"
                      className={`w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${darkMode
                        ? 'bg-gray-700/50 border-gray-600 text-white placeholder-gray-400'
                        : 'bg-gray-50 border-gray-200 placeholder-gray-400'
                        }`}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Subject</span>
                    <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    name="subject"
                    required
                    placeholder="What is this about?"
                    className={`w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${darkMode
                      ? 'bg-gray-700/50 border-gray-600 text-white placeholder-gray-400'
                      : 'bg-gray-50 border-gray-200 placeholder-gray-400'
                      }`}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Message</span>
                    <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    name="message"
                    required
                    rows={6}
                    placeholder="Your message here..."
                    className={`w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${darkMode
                      ? 'bg-gray-700/50 border-gray-600 text-white placeholder-gray-400'
                      : 'bg-gray-50 border-gray-200 placeholder-gray-400'
                      }`}
                  ></textarea>
                </div>

                <div className="flex items-center justify-between pt-4">
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <span className="text-red-500">*</span> Required fields
                  </p>
                  <button
                    type="submit"
                    className="px-8 py-3 rounded-xl font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 transform hover:scale-105 transition-all duration-200 shadow-lg"
                  >
                    Send Message
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className={`py-16 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} border-t ${darkMode ? 'border-gray-700' : 'border-gray-100'}`}>
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
            {/* Brand Section */}
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-3 group mb-6">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative bg-gradient-to-r from-indigo-600 to-blue-500 text-white p-2.5 rounded-xl shadow-lg">
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />
                    </svg>
                  </div>
                </div>
                <div className="flex flex-col">
                  <div className="text-2xl tracking-wide" style={{ fontFamily: '"Monomakh", serif' }}>
                    <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 via-indigo-500 to-blue-500">Find</span>
                    <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-blue-400 to-indigo-500">My</span>
                    <span className="bg-clip-text text-transparent bg-gradient-to-br from-indigo-500 via-blue-500 to-indigo-600">Angel</span>
                  </div>
                </div>
              </div>
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} max-w-xs text-sm leading-relaxed mb-6`}>
                Connecting ambitious startups with the right investors to fuel their growth and success.
              </p>
              <div className="flex space-x-4">
                <a href="#" className={`${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition duration-150`}>
                  <span className="sr-only">Twitter</span>
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                  </svg>
                </a>
                <a href="https://www.linkedin.com/company/findmyangel" className={`${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition duration-150`}>
                  <span className="sr-only">LinkedIn</span>
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                  </svg>
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Quick Links</h3>
              <ul className={`space-y-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                <li>
                  <button
                    onClick={() => {
                      const faqSection = document.getElementById('faq');
                      if (faqSection) faqSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }}
                    className="hover:text-indigo-500 transition duration-150"
                  >
                    FAQ
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => {
                      const pricingSection = document.getElementById('pricing');
                      if (pricingSection) pricingSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }}
                    className="hover:text-indigo-500 transition duration-150"
                  >
                    Pricing
                  </button>
                </li>
                <li>
                  <a href="/about" className="hover:text-indigo-500 transition duration-150">About Us</a>
                </li>
              </ul>
            </div>

            {/* Contact Info */}
            <div>
              <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Contact</h3>
              <ul className={`space-y-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  support@findmyangel.com
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Mumbai, India
                </li>
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className={`pt-8 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex flex-col md:flex-row justify-between items-center">
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} text-sm text-center md:text-left`}>
                © 2024 FindMyAngel. All rights reserved.
              </p>
              <div className={`mt-4 md:mt-0 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} flex flex-wrap items-center justify-center gap-1.5`}>
                <span>Made with</span>
                <svg className="w-4 h-4 text-red-500 inline-block" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                </svg>
                <span>by</span>
                <div className="flex items-center gap-1.5">
                  <a
                    href="https://www.linkedin.com/in/madhav0x1d"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-500 hover:text-indigo-600 transition-colors duration-200 hover:underline"
                  >
                    Madhav Shah
                  </a>
                  <span>&</span>
                  <a
                    href="https://www.linkedin.com/in/kush-tejani-42b109271"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-500 hover:text-indigo-600 transition-colors duration-200 hover:underline"
                  >
                    Kush Tejani
                  </a>
                </div>
              </div>
              <div className={`mt-4 md:mt-0 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                <a href="#" className="hover:text-indigo-500 transition duration-150">Privacy Policy</a>
                <span className="mx-2">·</span>
                <a href="#" className="hover:text-indigo-500 transition duration-150">Terms of Service</a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}