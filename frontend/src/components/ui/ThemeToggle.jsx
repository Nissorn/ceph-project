import React, { useEffect, useState } from 'react';

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Check initial preference
    if (typeof document !== 'undefined') {
      const isDarkMode = document.documentElement.classList.contains('dark') || 
                         localStorage.getItem('theme') === 'dark';
      
      if (isDarkMode) {
        document.documentElement.classList.add('dark');
        setIsDark(true);
      }
    }
  }, []);

  const toggleTheme = () => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark');
      const newTheme = !isDark;
      setIsDark(newTheme);
      localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    }
  };

  return (
    <button
      onClick={toggleTheme}
      className="px-4 py-2 text-sm font-medium rounded-lg bg-gray-200 hover:bg-gray-300 dark:bg-primary dark:hover:bg-opacity-80 text-gray-800 dark:text-gray-100 transition-colors"
      aria-label="Toggle Dark Mode"
    >
      {isDark ? '🌞 Light' : '🌙 Dark'}
    </button>
  );
}
