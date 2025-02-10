import { Link } from 'react-router-dom'
import "./Navbar_style.css";
import logo from "../assets/FPL-Master_Logo.png"; // Adjust the path based on where Navbar_page.jsx is

const Navbar = () => {
  return (
    <nav className="navbar">
      <img src={logo} alt="FPL-Master Logo" className="navbar-logo" />
      <div className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/fixtures">Fixtures</Link>
        <Link to="/previous-games">Previous Games</Link>
        <Link to="/players">Players</Link>
        <Link to="/contact">Contact</Link>
      </div>
    </nav>
  );
};

export default Navbar;
