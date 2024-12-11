import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <img src="/Users/nayeb/Desktop/FPL-MASTER/frontend/src/assets/FPL-Master_Logo.png" alt="logo" />
          Home Page
        
      </div>
      <div className="navbar-center">
        <ul className="nav-links">
          <li>
            <Link to="/fixtures">Fixtures</Link>
          </li>
          <li>
            <Link to="/players">All Players</Link>
          </li>
          <li>
            <Link to="/contact">Contact-Us</Link>
          </li>
        </ul>
      </div>
      
    </nav>
  );
};

export default Navbar;
