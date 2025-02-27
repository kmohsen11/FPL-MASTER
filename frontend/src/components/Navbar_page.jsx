import { useState } from 'react';
import { Link } from 'react-router-dom'
import "./Navbar_style.css";
import logo from "../assets/FPL-Master_Logo.png"; // Adjust the path based on where Navbar_page.jsx is

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };
  
  return (
    <nav className="navbar">
      <img src={logo} alt="FPL-Master Logo" className="navbar-logo" />
      <div className="menu-toggle" onClick={toggleMenu}>
        <span></span>
        <span></span>
        <span></span>
      </div>
      <div className={`navbar-links ${isMenuOpen ? "active" : ""}`}>
        <Link to="/" onClick={() => setIsMenuOpen(false)}>
          Home
        </Link>
        <Link to="/fixtures" onClick={() => setIsMenuOpen(false)}>
          Fixtures
        </Link>
        <Link to="/previous-games" onClick={() => setIsMenuOpen(false)}>
          Previous Games
        </Link>
        <Link to="/players" onClick={() => setIsMenuOpen(false)}>
          Players
        </Link>
        <Link to="/contact" onClick={() => setIsMenuOpen(false)}>
          Contact
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
