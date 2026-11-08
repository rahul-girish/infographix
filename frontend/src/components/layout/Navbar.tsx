
import { useState } from "react";
import { Link } from "react-router-dom";
import { Atom, ArrowUpRight, Menu, X } from "lucide-react";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/95 backdrop-blur-xl">
      <nav className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6">

        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 text-white">
            <Atom size={26} />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-slate-900">
              Infographix
            </h1>
            <p className="text-[10px] uppercase tracking-widest text-slate-500">
              Research to Visuals
            </p>
          </div>
        </Link>

        <div className="hidden items-center gap-8 text-sm font-medium text-slate-600 md:flex">
          <a href="/#features" className="hover:text-blue-600">Features</a>
          <a href="/#workflow" className="hover:text-blue-600">How it works</a>
          <a href="/#templates" className="hover:text-blue-600">Templates</a>
          <a href="/#about" className="hover:text-blue-600">About</a>
        </div>

        <div className="hidden items-center gap-5 md:flex">
          <Link to="/login" className="font-semibold text-slate-700 hover:text-blue-600">
            Log in
          </Link>
          <Link to="/register" className="flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 font-semibold text-white transition hover:bg-blue-700">
            Get Started <ArrowUpRight size={17} />
          </Link>
        </div>

        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="rounded-lg p-2 text-slate-700 md:hidden"
          aria-label="Toggle menu"
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X /> : <Menu />}
        </button>
      </nav>

      {menuOpen && (
        <div className="space-y-5 border-t border-slate-100 bg-white p-6 md:hidden">
          <a href="/#features" onClick={() => setMenuOpen(false)} className="block">Features</a>
          <a href="/#workflow" onClick={() => setMenuOpen(false)} className="block">How it works</a>
          <a href="/#templates" onClick={() => setMenuOpen(false)} className="block">Templates</a>
          <Link to="/login" className="block">Log in</Link>
          <Link to="/register" className="block rounded-lg bg-blue-600 p-3 text-center text-white">
            Get Started
          </Link>
        </div>
      )}
    </header>
  );
}
