import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <Link to="/" className="logo">
          Home Page
        </Link>
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
