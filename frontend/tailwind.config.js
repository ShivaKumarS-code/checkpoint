/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#07090d",
          900: "#0b0e14",
          850: "#10141c",
          800: "#151a24",
          700: "#222937",
          600: "#354052",
          500: "#4b586d",
          400: "#7d8ba0",
          300: "#aab5c5",
          200: "#cbd3df",
          100: "#eef2f8"
        },
        accent: {
          DEFAULT: "#62d9e8",
          soft: "#1a4149",
          dim: "#8fe8f2"
        },
        good: "#5ee2a0",
        bad: "#ff7b93",
        warn: "#f2c66d",
        violet: "#b49cff"
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["Cascadia Code", "JetBrains Mono", "Consolas", "monospace"]
      },
      boxShadow: {
        panel: "0 18px 50px rgba(0, 0, 0, 0.35)"
      }
    }
  },
  plugins: []
};
