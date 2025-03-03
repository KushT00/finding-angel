/* eslint-disable @next/next/no-img-element */
import { useState } from "react";

// Define TypeScript interfaces for the component props
interface User {
  picture?: string;
  // Add other user properties as needed
}

interface NavbarProps {
  user: User | null;
  handleSignIn: () => void;
  handleSignOut: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}

const Navbar = ({ user, handleSignIn, handleSignOut, darkMode, toggleDarkMode }: NavbarProps) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 ${
        darkMode ? "bg-gray-900/80" : "bg-white/80"
      } backdrop-blur-sm border-b ${darkMode ? "border-gray-800" : "border-gray-100"}`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="text-2xl font-bold">Find My Angel</div>
          </div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-4">
            {!user ? (
              /* Sign In Button (Only when not logged in) */
              <button
                onClick={handleSignIn}
                className="px-4 py-2 text-white bg-blue-600 hover:bg-blue-700 transition rounded-lg font-medium shadow-sm"
              >
                Login
              </button>
            ) : (
              /* Theme Toggle and Profile (Only when logged in) */
              <>
                {/* Theme Toggle */}
                <button
                  onClick={toggleDarkMode}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition"
                >
                  {darkMode ? (
                    <svg className="w-5 h-5 text-gray-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                      />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-gray-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                      />
                    </svg>
                  )}
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
                      className="w-8 h-8 rounded-full"
                    />
                  </button>

                  {dropdownOpen && (
                    <div className="absolute right-0 mt-2 w-40 bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-md shadow-lg z-10">
                      <button
                        onClick={handleSignOut}
                        className="w-full px-4 py-2 text-left text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                      >
                        Sign Out
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;