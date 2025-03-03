/* eslint-disable react/no-unescaped-entities */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useDarkMode } from '../../../context/DarkModeContext';
import { auth } from '../../../lib/firebase';
import {
  ChevronLeft,
  Building,
  MapPin,
  Target,
  Calendar,
  DollarSign,
  Users,
  Briefcase,
  LineChart,
  Globe,
  Mail,
  Award,
  TrendingUp,
  Info,
  Hash,
  FileText,
  Building2,
  Sparkles,
  Share2,
  Download
} from 'lucide-react';
import Link from 'next/link';

// Helper function to handle missing values
const formatValue = (value: any, type: 'number' | 'string' | 'array' = 'string') => {
  if (value === undefined || value === null) {
    return type === 'array' ? [] : type === 'number' ? 0 : 'N/A';
  }
  return value;
};

const formatWebsiteUrl = (url: string) => {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }
  return `https://${url}`;
};

export default function InvestorProfile() {
  const { darkMode } = useDarkMode();
  const router = useRouter();
  const params = useParams();
  const [investor, setInvestor] = useState<any>(null);
  const [description, setDescription] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [aiSummary, setAiSummary] = useState<string>('');
  const [isLoadingSummary, setIsLoadingSummary] = useState(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [showDownloadReminder, setShowDownloadReminder] = useState(false);
  const [showShareToast, setShowShareToast] = useState(false);

  useEffect(() => {
    // Show download reminder toast after a short delay
    const timer = setTimeout(() => {
      setShowDownloadReminder(true);
      // Hide the reminder after 10 seconds (increased from 5 to 10)
      setTimeout(() => setShowDownloadReminder(false), 10000);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  // Move fetchAISummary outside useEffect so it can be called by retry button
  const fetchAISummary = async () => {
    setIsLoadingSummary(true);
    setSummaryError(null);

    try {
      const user = auth.currentUser;
      if (!user) {
        setSummaryError("Authentication required");
        return;
      }

      if (!params.id) {
        throw new Error("Invalid investor ID");
      }

      const token = await user.getIdToken();
      const cleanId = params.id
        ?.toString()
        .replace('http://', '')
        .replace('https://', '')
        .replace('www.', '');

      console.log("Fetching AI summary for ID:", cleanId); // Debug log

      const response = await fetch(`https://findmyangelapi.vercel.app/api/investors/${cleanId}/ai-summary`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      console.log("AI Summary Response status:", response.status); // Debug log

      const data = await response.json();
      console.log("AI Summary Data:", data); // Debug log

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch AI summary');
      }

      if (!data.ai_summary) {
        throw new Error('No summary received from server');
      }

      setAiSummary(data.ai_summary);
      setSummaryError(null);
    } catch (error: any) {
      console.error('Error fetching AI summary:', error);
      setSummaryError(error.message || 'Unable to generate AI summary. Please try again later.');
    } finally {
      setIsLoadingSummary(false);
    }
  };


  useEffect(() => {
    const fetchData = async () => {
      try {
        const user = auth.currentUser;
        if (!user) return;

        const token = await user.getIdToken();
        const headers = {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        };

        if (!params.id) {
          throw new Error("Invalid investor ID");
        }

        const cleanId = params.id
          ?.toString()
          .replace('http://', '')
          .replace('https://', '')
          .replace('www.', '');

        // Use the regular endpoint since we want full data
        const response = await fetch(`https://findmyangelapi.vercel.app/api/investors/${cleanId}`, {
          headers,
          cache: 'no-store'
        });

        if (!response.ok) {
          if (response.status === 403) {
            router.push('/credits');
            return;
          }
          throw new Error(`Failed to fetch data`);
        }

        const data = await response.json();
        setInvestor(data.investor);
        setDescription(data.description);

      } catch (error) {
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };

    if (params.id) {
      fetchData().then(() => {
        // Only fetch AI summary after basic data is loaded
        fetchAISummary();
      });
    }
  }, [fetchAISummary, params.id, router]);

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `${investor.name} - Investor Profile`,
          text: `Check out ${investor.name}'s investor profile`,
          url: window.location.href,
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback: Copy to clipboard
      navigator.clipboard.writeText(window.location.href);
      setShowShareToast(true);
      setTimeout(() => setShowShareToast(false), 3000);
    }
  };

  const handleDownload = async () => {
    try {
      setLoading(true);
      const user = auth.currentUser;
      if (!user) {
        console.error('User not authenticated');
        return;
      }

      const token = await user.getIdToken();
      const response = await fetch(`https://findmyangelapi.vercel.app/api/investors/${params.id}/download`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `${investor?.name || 'investor'}_profile.csv`;

      document.body.appendChild(a);
      a.click();

      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading profile:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!investor) {
    return (
      <div className={`min-h-screen flex flex-col items-center justify-center ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'}`}>
        <h1 className="text-2xl mb-4">Investor not found</h1>
        <Link href="/dashboard" className="text-blue-500 hover:text-blue-400">
          Return to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-50'} transition-colors duration-200`}>
      <div className={`w-full ${darkMode ? 'bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900' : 'bg-gradient-to-r from-blue-50 via-white to-blue-50'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-12">
          {/* Header Buttons */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
            <button
              onClick={() => router.back()}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${darkMode
                ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                : 'bg-white text-gray-600 hover:bg-gray-50'
                } shadow-sm hover:shadow-md transition-all duration-200 text-sm md:text-base`}
            >
              <ChevronLeft size={16} />
              Back to Dashboard
            </button>

            <div className="flex flex-col md:flex-row gap-2 w-full md:w-auto">
              <button
                onClick={handleShare}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${darkMode
                  ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
                  } shadow-sm hover:shadow-md transition-all duration-200 text-sm md:text-base w-full md:w-auto`}
              >
                <Share2 size={16} />
                Share
              </button>

              <button
                onClick={handleDownload}
                disabled={loading}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${darkMode
                  ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
                  } shadow-sm hover:shadow-md transition-all duration-200 text-sm md:text-base w-full md:w-auto ${loading ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
              >
                <div className="flex items-center gap-2">
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-gray-300 border-t-transparent rounded-full animate-spin"></div>
                      <span>Downloading...</span>
                    </>
                  ) : (
                    <>
                      <Download size={16} />
                      <span>Download Profile</span>
                    </>
                  )}
                </div>
              </button>
            </div>
          </div>

          {/* Title and Subtitle */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
            <div className="space-y-4">
              <div className="transform transition-all duration-200 hover:translate-x-2">
                <h1 className={`text-3xl md:text-5xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'} tracking-tight`}>
                  {formatValue(investor.name)}
                </h1>
                <p className={`text-lg md:text-xl mt-3 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  {formatValue(investor.company_name)}
                </p>
              </div>

              {/* Quick Info Pills */}
              <div className="flex flex-wrap gap-3 mt-4">
                {investor.Fund_Type && (
                  <span className={`px-4 py-1.5 rounded-full text-sm font-medium ${darkMode ? 'bg-blue-900/30 text-blue-200' : 'bg-blue-100 text-blue-800'
                    }`}>
                    {formatValue(investor.Fund_Type)}
                  </span>
                )}
                {investor.location?.[0] && (
                  <span className={`px-4 py-1.5 rounded-full text-sm font-medium ${darkMode ? 'bg-purple-900/30 text-purple-200' : 'bg-purple-100 text-purple-800'
                    }`}>
                    <MapPin size={14} className="inline mr-1" />
                    {formatValue(investor.location[0])}
                  </span>
                )}
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 w-full md:w-auto">
              {[
                {
                  value: formatValue(investor.No_Of_Investments, 'number'),
                  label: 'Investments',
                  icon: <Target size={20} />,
                  color: 'blue'
                },
                {
                  value: formatValue(investor.No_Of_Exits, 'number'),
                  label: 'Exits',
                  icon: <TrendingUp size={20} />,
                  color: 'green'
                },
                {
                  value: formatValue(investor.Founding_Year, 'number'),
                  label: 'Founded',
                  icon: <Calendar size={20} />,
                  color: 'purple'
                }
              ].map((stat, index) => (
                <div
                  key={index}
                  className={`p-4 md:p-6 rounded-2xl transform transition-all duration-200 
            hover:scale-105 hover:shadow-xl ${darkMode
                      ? 'bg-gray-800 text-gray-300 hover:bg-gray-750'
                      : 'bg-white text-gray-600 hover:bg-gray-50'
                    } shadow-lg`}
                >
                  <div className={`text-${stat.color}-500 mb-2`}>{stat.icon}</div>
                  <p className={`text-2xl md:text-3xl font-bold text-${stat.color}-500 mb-1`}>
                    {stat.value}
                  </p>
                  <p className="text-sm opacity-80">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content with Enhanced Cards */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* AI Summary Card */}
        <div className={`mb-8 p-8 rounded-2xl shadow-lg transform transition-all duration-200 
          hover:shadow-xl ${darkMode ? 'bg-gray-800/80' : 'bg-white'}`}>
          <div className="flex items-center gap-3 mb-6">
            <div className={`p-3 rounded-xl ${darkMode ? 'bg-purple-900/30' : 'bg-purple-100'
              }`}>
              <Sparkles className="text-purple-500" size={24} />
            </div>
            <h2 className="text-2xl font-bold">AI-Generated Summary</h2>
          </div>

          {isLoadingSummary ? (
            <div className="space-y-3">
              <div className="flex items-center space-x-4 animate-pulse">
                <div className="h-4 bg-gray-300 rounded w-3/4"></div>
              </div>
              <div className="flex items-center space-x-4 animate-pulse">
                <div className="h-4 bg-gray-300 rounded w-full"></div>
              </div>
              <div className="flex items-center space-x-4 animate-pulse">
                <div className="h-4 bg-gray-300 rounded w-2/3"></div>
              </div>
            </div>
          ) : summaryError ? (
            <div className="flex items-center gap-2 text-red-500">
              <span className="text-lg">{summaryError}</span>
              <button
                onClick={() => {
                  if (params.id) {
                    fetchAISummary();
                  }
                }}
                className="px-3 py-1 text-sm rounded-full bg-red-100 text-red-600 hover:bg-red-200 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : (
            <div className={`relative ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <div className="absolute -left-4 top-0 text-4xl opacity-20">&quot;</div>
              <p className="text-lg leading-relaxed pl-4">
                {aiSummary}
              </p>
              <div className="absolute -right-4 bottom-0 text-4xl opacity-20">&quot;</div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* About Section */}
            <div className={`p-8 rounded-2xl shadow-lg transform transition-all duration-200 
              hover:shadow-xl ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex items-center gap-3 mb-6">
                <div className={`p-3 rounded-xl ${darkMode ? 'bg-blue-900/30' : 'bg-blue-100'}`}>
                  <Info className="text-blue-500" size={24} />
                </div>
                <h2 className="text-2xl font-bold">About</h2>
              </div>
              <p className={`text-lg leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {formatValue(investor.headline)}
              </p>
            </div>

            {/* Investment Focus */}
            <div className={`p-8 rounded-2xl shadow-lg transform transition-all duration-200 
              hover:shadow-xl ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex items-center gap-3 mb-8">
                <div className={`p-3 rounded-xl ${darkMode ? 'bg-green-900/30' : 'bg-green-100'}`}>
                  <Target className="text-green-500" size={24} />
                </div>
                <h2 className="text-2xl font-bold">Investment Focus</h2>
              </div>

              <div className="space-y-8">
                {/* Investment Stages */}
                {formatValue(investor.stage, 'array').length > 0 && (
                  <div>
                    <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      Investment Stages
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {formatValue(investor.stage, 'array').map((stage: string, index: number) => (
                        <span
                          key={index}
                          className={`px-4 py-2 rounded-full text-sm font-medium ${darkMode ? 'bg-blue-900/30 text-blue-200' : 'bg-blue-100 text-blue-800'
                            }`}
                        >
                          {stage}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Focus Areas */}
                {formatValue(investor.fund_focus, 'array').length > 0 && (
                  <div>
                    <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      Focus Areas
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {formatValue(investor.fund_focus, 'array').map((focus: string, index: number) => (
                        <span
                          key={index}
                          className={`px-4 py-2 rounded-full text-sm font-medium ${darkMode ? 'bg-green-900/30 text-green-200' : 'bg-green-100 text-green-800'
                            }`}
                        >
                          {focus}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Portfolio Companies */}
            {formatValue(investor.portfolio_Companies, 'array').length > 0 && (
              <div className={`p-8 rounded-2xl shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex items-center gap-3 mb-8">
                  <div className={`p-3 rounded-xl ${darkMode ? 'bg-purple-900/30' : 'bg-purple-100'}`}>
                    <Building2 className="text-purple-500" size={24} />
                  </div>
                  <h2 className="text-2xl font-bold">Portfolio Companies</h2>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {formatValue(investor.portfolio_Companies, 'array').map((company: { foreignRowDisplayName: string }, index: number) => (
                    <div
                      key={index}
                      className={`p-4 rounded-xl transform transition-all duration-200 
                        hover:scale-105 hover:shadow-lg ${darkMode
                          ? 'bg-gray-700/50 text-gray-200 hover:bg-gray-700'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                        } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                    >
                      <p className="font-medium">{formatValue(company.foreignRowDisplayName)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Contact Card */}
            {(formatValue(investor.website) || formatValue(investor.email)) && (
              <div className={`p-8 rounded-2xl shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex items-center gap-3 mb-6">
                  <div className={`p-3 rounded-xl ${darkMode ? 'bg-indigo-900/30' : 'bg-indigo-100'}`}>
                    <Mail className="text-indigo-500" size={24} />
                  </div>
                  <h2 className="text-2xl font-bold">Contact</h2>
                </div>

                <div className="space-y-4">
                  {formatValue(investor.website) && (
                    <a
                      href={formatWebsiteUrl(investor.website)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`flex items-center gap-3 p-4 rounded-xl transform transition-all duration-200 
                        hover:scale-102 hover:shadow-md ${darkMode
                          ? 'bg-gray-700/50 hover:bg-gray-700 text-gray-200'
                          : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
                        } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                    >
                      <Globe size={18} className="text-blue-500" />
                      <span className="text-sm">{formatValue(investor.website)}</span>
                    </a>
                  )}

                  {formatValue(investor.email) && (
                    <a
                      href={`mailto:${formatValue(investor.email)}`}
                      className={`flex items-center gap-3 p-4 rounded-xl transform transition-all duration-200 
                        hover:scale-102 hover:shadow-md ${darkMode
                          ? 'bg-gray-700/50 hover:bg-gray-700 text-gray-200'
                          : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
                        } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                    >
                      <Mail size={18} className="text-blue-500" />
                      <span className="text-sm">{formatValue(investor.email)}</span>
                    </a>
                  )}
                </div>
              </div>
            )}

            {/* Social Media */}
            <div className={`p-8 rounded-2xl shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex items-center gap-3 mb-6">
                <div className={`p-3 rounded-xl ${darkMode ? 'bg-pink-900/30' : 'bg-pink-100'}`}>
                  <Globe className="text-pink-500" size={24} />
                </div>
                <h2 className="text-2xl font-bold">Social Media</h2>
              </div>

              <div className="space-y-4">
                {investor.Linkedin && (
                  <a
                    href={investor.Linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`flex items-center gap-3 p-4 rounded-xl transform transition-all duration-200 
                      hover:scale-102 hover:shadow-md ${darkMode
                        ? 'bg-gray-700/50 hover:bg-gray-700 text-gray-200'
                        : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
                      } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                  >
                    <div className={`p-2 rounded-lg ${darkMode ? 'bg-blue-900/30' : 'bg-blue-100'}`}>
                      <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                      </svg>
                    </div>
                    <span className="font-medium">LinkedIn Profile</span>
                  </a>
                )}

                {investor.twitter && (
                  <a
                    href={investor.twitter}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`flex items-center gap-3 p-4 rounded-xl transform transition-all duration-200 
                      hover:scale-102 hover:shadow-md ${darkMode
                        ? 'bg-gray-700/50 hover:bg-gray-700 text-gray-200'
                        : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
                      } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                  >
                    <div className={`p-2 rounded-lg ${darkMode ? 'bg-blue-900/30' : 'bg-blue-100'}`}>
                      <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
                      </svg>
                    </div>
                    <span className="font-medium">Twitter Profile</span>
                  </a>
                )}

                {investor.Facebook_Link && (
                  <a
                    href={investor.Facebook_Link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`flex items-center gap-3 p-4 rounded-xl transform transition-all duration-200 
                      hover:scale-102 hover:shadow-md ${darkMode
                        ? 'bg-gray-700/50 hover:bg-gray-700 text-gray-200'
                        : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
                      } border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}
                  >
                    <div className={`p-2 rounded-lg ${darkMode ? 'bg-blue-900/30' : 'bg-blue-100'}`}>
                      <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                      </svg>
                    </div>
                    <span className="font-medium">Facebook Profile</span>
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Share Success Toast */}
      {showShareToast && (
        <div className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg transform transition-all duration-300 
          ${darkMode
            ? 'bg-green-900/90 text-white'
            : 'bg-green-50 text-green-900'
          } border ${darkMode ? 'border-green-700' : 'border-green-200'}`}
        >
          <div className="flex items-center gap-3">
            <svg
              className={`w-5 h-5 ${darkMode ? 'text-green-300' : 'text-green-600'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            <div>
              <p className="font-medium">Link copied!</p>
              <p className={`text-sm ${darkMode ? 'text-green-200' : 'text-green-700'}`}>
                Profile URL copied to clipboard
              </p>
            </div>
            <button
              onClick={() => setShowShareToast(false)}
              className={`ml-4 p-1 rounded-full 
                ${darkMode
                  ? 'hover:bg-green-800 text-green-200'
                  : 'hover:bg-green-100 text-green-600'
                } transition-colors`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Download Reminder Toast */}
      {showDownloadReminder && (
        <div className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg transform transition-all duration-300 
          ${darkMode
            ? 'bg-blue-900/90 text-white'
            : 'bg-blue-50 text-blue-900'
          } border ${darkMode ? 'border-blue-700' : 'border-blue-200'}`}
        >
          <div className="flex items-center gap-3">
            <Download size={20} className={`${darkMode ? 'text-blue-300' : 'text-blue-600'}`} />
            <div>
              <p className="font-medium">Don't forget!</p>
              <p className={`text-sm ${darkMode ? 'text-blue-200' : 'text-blue-700'}`}>
                You can download this investor's data for offline use.
              </p>
            </div>
            <button
              onClick={() => setShowDownloadReminder(false)}
              className={`ml-4 p-1 rounded-full 
                ${darkMode
                  ? 'hover:bg-blue-800 text-blue-200'
                  : 'hover:bg-blue-100 text-blue-600'
                } transition-colors`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 