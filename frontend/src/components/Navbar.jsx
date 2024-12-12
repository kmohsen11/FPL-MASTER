import { Link } from 'react-router-dom'
import './Navbar.css'

const Navbar = () => {
  return (
    <nav className="navbar">
      <img
        src="./src/assets/FPL-Master_Logo.png"
        alt="FPL-Master Logo"
        className="navbar-logo"
      />
      <div className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/fixtures">Fixtures</Link>
        <Link to="/previous-games">Previous Games</Link>
        <Link to="/players">Players</Link>
        <Link to="/contact">Contact</Link>
      </div>
    </nav>
  )
}

export default Navbar
