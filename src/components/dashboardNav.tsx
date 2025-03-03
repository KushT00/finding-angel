/* eslint-disable @next/next/no-img-element */
import { Sun, Moon } from "lucide-react";
import { useState } from "react";

// Define TypeScript interfaces for the component props
export interface Stats {
  total_investors?: number;
  total_industries?: number;
  total_locations?: number;
}

interface Credits {
  total: number;
}

interface User {
  picture?: string;
}

interface DashboardNavbarProps {
  Stats: Stats | null;
  credits: Credits;
  darkMode: boolean;
  toggleDarkMode: () => void;
  handleJoinCommunity: () => void;
  handleGetMoreCredits: () => void;
  handleLogout: () => void;
  user?: User;
}

const DashboardNavbar = ({
  Stats,
  credits,
  darkMode,
  toggleDarkMode,
  handleJoinCommunity,
  handleGetMoreCredits,
  handleLogout,
  user
}: DashboardNavbarProps) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="mb-10">
      {/* Mobile view (stacked layout) - Only visible on small screens */}
      <div className="md:hidden">
        {/* Top bar with logo and profile for mobile */}
        <div className="flex justify-between items-center mb-5">
          {/* Logo */}
          <div className="flex items-center space-x-3 group">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition duration-300"></div>

            </div>
            <div className="flex flex-col">
              <div className="text-xl tracking-wide" style={{ fontFamily: '"Monomakh", serif' }}>
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
                <div className="h-[1px] w-3 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
                <span className="text-[8px] font-medium tracking-widest text-indigo-600 dark:text-indigo-400" style={{ fontFamily: "'Inter', sans-serif" }}>
                  INVESTOR NETWORK
                </span>
                <div className="h-[1px] w-3 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
              </div>
            </div>
          </div>

          {/* User Profile and Theme Toggle */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleDarkMode}
              className={`p-2 rounded-full border ${darkMode
                  ? 'bg-gray-800 text-gray-200 border-gray-700'
                  : 'bg-gray-100 text-gray-600 border-gray-300 shadow-sm hover:bg-gray-200'
                } transition-all duration-200`}
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>

            {/* Profile Dropdown */}
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center space-x-2"
                aria-expanded={dropdownOpen}
                aria-haspopup="true"
              >
                <img
                  src={user?.picture || "https://github.com/shadcn.png"}
                  alt="User profile"
                  className="w-9 h-9 rounded-full border-2 border-indigo-500 hover:border-indigo-600 transition-colors"
                />
              </button>

              {/* Dropdown Menu */}
              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 py-2 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-10">
                  <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                    User Profile
                  </div>
                  <button
                    onClick={() => {
                      setDropdownOpen(false);
                      handleLogout();
                    }}
                    className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Stats and Action Buttons for mobile */}
        <div className="flex flex-col space-y-4">

          {/* Action Buttons */}
          <div className="flex w-full gap-3">
            {/* Container for both buttons with equal width */}
            <div className="flex w-full">
              {/* Community/Credits buttons - now in a grid with equal width */}
              <div className="grid grid-cols-2 gap-3 w-full">
                {credits.total > 0 ? (
                  <button
                    onClick={handleJoinCommunity}
                    className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-4 py-2 rounded-lg 
                transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-105 flex items-center justify-center gap-2 text-sm w-full"
                  >
                    <svg
                      className="w-4 h-4"
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
                    <span className="whitespace-nowrap">Join Community</span>
                  </button>
                ) : (
                  <button
                    onClick={handleGetMoreCredits}
                    className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-4 py-2 rounded-lg 
                transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-105 flex items-center justify-center gap-2 text-sm w-full"
                  >
                    <svg
                      className="w-4 h-4"
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
                    <span className="whitespace-nowrap">Get Credits</span>
                  </button>
                )}

                {/* Credits Counter - now with full width to match buttons */}
                <div className={`flex items-center justify-center gap-2 px-3 py-2 rounded-lg border w-full ${darkMode
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
                  <span className={`font-medium ${darkMode ? 'text-indigo-400' : 'text-indigo-600'
                    }`}>
                    {credits.total}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap gap-4">
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_investors || 0}</span>
              <span>Investors</span>
            </div>
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_industries || 0}</span>
              <span>Industries</span>
            </div>
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_locations || 0}</span>
              <span>Locations</span>
            </div>
          </div>
        </div>
      </div>

      {/* Desktop view (single line) - Only visible on medium screens and up */}
      <div className="hidden md:flex justify-between items-center">
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
                  <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />
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
                <span className="text-[9px] font-medium tracking-widest text-indigo-600 dark:text-indigo-400" style={{ fontFamily: "'Inter', sans-serif" }}>
                  INVESTOR NETWORK
                </span>
                <div className="h-[1px] w-4 bg-gradient-to-r from-transparent via-indigo-200 to-transparent dark:via-indigo-800"></div>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="flex gap-8">
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_investors || 0}</span>
              <span>Investors</span>
            </div>
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_industries || 0}</span>
              <span>Industries</span>
            </div>
            <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} flex items-center gap-2`}>
              <span className="font-medium text-indigo-600">{Stats?.total_locations || 0}</span>
              <span>Locations</span>
            </div>
          </div>
        </div>

        {/* Right side actions */}
        <div className="flex items-center gap-4">
          {/* Community Button */}
          {credits.total > 0 && (
            <button
              onClick={handleJoinCommunity}
              className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-6 py-2 rounded-lg 
                transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-105 flex items-center gap-2"
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
              <span>Join Community</span>
            </button>
          )}
          {/* Get Credits Button */}
          {credits.total === 0 && (
            <button
              onClick={handleGetMoreCredits}
              className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-6 py-2 rounded-lg 
                transition-all duration-200 shadow-sm hover:shadow-lg hover:scale-105 flex items-center gap-2"
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
              <span>Get Credits</span>
            </button>
          )}

          {/* Credits Counter */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${darkMode
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
            <span className={`font-medium ${darkMode ? 'text-indigo-400' : 'text-indigo-600'
              }`}>
              {credits.total}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleDarkMode}
            className={`p-2.5 rounded-lg border ${darkMode
                ? 'bg-gray-800 text-gray-200 border-gray-700'
                : 'bg-gray-100 text-gray-600 border-gray-300 shadow-sm hover:bg-gray-200'
              } transition-all duration-200`}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          {/* Profile Dropdown */}
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center space-x-2"
              aria-expanded={dropdownOpen}
              aria-haspopup="true"
            >
              <img
                src={user?.picture || "https://github.com/shadcn.png"}
                alt="User profile"
                className="w-9 h-9 rounded-full border-2 border-indigo-500 hover:border-indigo-600 transition-colors"
              />
            </button>

            {/* Dropdown Menu */}
            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-48 py-2 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-10">
                <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  User Profile
                </div>
                <button
                  onClick={() => {
                    setDropdownOpen(false);
                    handleLogout();
                  }}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardNavbar;