'use client';
import { useState } from 'react';
import Link from 'next/link';

export default function AboutUs() {
  const [darkMode, setDarkMode] = useState(true); // Set to true by default for dark mode

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gradient-to-b from-gray-50 to-white'}`}>
      {/* Navbar */}
      <nav className={`fixed top-0 left-0 right-0 z-50 ${darkMode ? 'bg-gray-900/80' : 'bg-white/80'} backdrop-blur-sm border-b ${darkMode ? 'border-gray-800' : 'border-gray-100'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Left side: Logo */}
            <Link href="/" className="flex items-center space-x-3 group">
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
                    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
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
            </Link>

            {/* Right side: Navigation Links and Dark Mode Toggle */}
            <div className="flex items-center space-x-8">
              <div className="hidden md:flex items-center space-x-8">
                <Link href="/" className={`${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition-colors duration-200`}>
                  Home
                </Link>
                <Link href="/about" className={`${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition-colors duration-200`}>
                  About
                </Link>
              </div>
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-100'} transition-colors duration-200`}
                aria-label="Toggle dark mode"
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
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className={`relative overflow-hidden pt-16 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:w-full lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div className="text-center">
                <h1 className={`text-4xl tracking-tight font-extrabold sm:text-5xl md:text-6xl ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  <span className="block xl:inline">About</span>{' '}
                  <span className="block text-indigo-600 xl:inline">FindMyAngel</span>
                </h1>
                <p className={`mt-3 max-w-md mx-auto text-base sm:text-lg md:mt-5 md:text-xl md:max-w-3xl ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                  Empowering startups to connect with the right investors, making fundraising more accessible and efficient.
                </p>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Mission Section */}
      <div className={`py-16 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className={`text-base text-indigo-600 font-semibold tracking-wide uppercase ${darkMode ? 'text-indigo-400' : ''}`}>Our Mission</h2>
            <p className={`mt-2 text-3xl leading-8 font-extrabold tracking-tight ${darkMode ? 'text-white' : 'text-gray-900'} sm:text-4xl`}>
              Revolutionizing Startup Funding
            </p>
            <p className={`mt-4 max-w-2xl text-xl ${darkMode ? 'text-gray-300' : 'text-gray-500'} lg:mx-auto`}>
              We're building a bridge between innovative startups and visionary investors, creating opportunities for growth and success.
            </p>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className={`py-12 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {/* Feature 1 */}
            <div className={`rounded-lg p-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
              <div className="w-12 h-12 bg-indigo-500 rounded-md flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Fast & Efficient</h3>
              <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                Our platform streamlines the fundraising process, saving valuable time for both startups and investors.
              </p>
            </div>

            {/* Feature 2 */}
            <div className={`rounded-lg p-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
              <div className="w-12 h-12 bg-indigo-500 rounded-md flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              </div>
              <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Verified Network</h3>
              <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                Access our carefully curated network of verified investors and startups.
              </p>
            </div>

            {/* Feature 3 */}
            <div className={`rounded-lg p-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
              <div className="w-12 h-12 bg-indigo-500 rounded-md flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Secure Platform</h3>
              <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                Your data and interactions are protected with enterprise-grade security.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Team Section */}
      <div className={`py-16 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className={`text-3xl font-extrabold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Meet Our Team
            </h2>
            <p className={`mt-4 max-w-2xl mx-auto text-xl ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
              The minds behind FindMyAngel
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
              {/* Madhav Shah */}
              <div className={`text-center ${darkMode ? 'bg-gray-900' : 'bg-white'} rounded-lg p-8 shadow-lg`}>
                <img
                  className="mx-auto h-32 w-32 rounded-full border-4 border-indigo-500"
                  src="/madhav.jpeg"
                  alt="Madhav Shah"
                />
                <h3 className={`mt-6 text-xl font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Madhav Shah
                </h3>
                <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>Co-Founder & Developer</p>
                <div className="mt-4">
                  <a
                    href="https://www.linkedin.com/in/madhav0x1d"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-500 hover:text-indigo-600 transition-colors duration-200"
                  >
                    Connect on LinkedIn
                  </a>
                </div>
              </div>

              {/* Kush Tejani */}
              <div className={`text-center ${darkMode ? 'bg-gray-900' : 'bg-white'} rounded-lg p-8 shadow-lg`}>
                <img
                  className="mx-auto h-32 w-32 rounded-full border-4 border-indigo-500"
                  src="/kushtejani.jpeg"
                  alt="Kush Tejani"
                />
                <h3 className={`mt-6 text-xl font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Kush Tejani
                </h3>
                <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>Co-Founder & Developer</p>
                <div className="mt-4">
                  <a
                    href="https://www.linkedin.com/in/kush-tejani-42b109271"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-500 hover:text-indigo-600 transition-colors duration-200"
                  >
                    Connect on LinkedIn
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className={`${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8">
          <div className={`p-8 rounded-2xl shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
            <h2 className={`text-3xl font-extrabold text-center ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Ready to Get Started?
            </h2>
            <p className={`mt-4 text-lg text-center ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
              Join FindMyAngel today and take the first step towards securing funding for your startup.
            </p>
            <div className="mt-8 flex justify-center">
              <Link
                href="/signup"
                className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10 transition-all duration-200"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 