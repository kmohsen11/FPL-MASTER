import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./components/Home";
import Fixtures from "./components/Fixtures";
import PreviousGames from "./components/PreviousGames";
import Players from "./components/Players";
import Contact from "./components/Contact";
import "./App.css";

function App() {
  return (
    <Router>
      <div>
        <nav className="navbar">
          <img src="./src/assets/FPL-Master_Logo.png" alt= "FPL-Master Logo" className="navbar-logo" />
          <div className="nav-links">
            <Link to="/">Home</Link>
            <Link to="/fixtures">Fixtures</Link>
            <Link to="/previous-games">Previous Games</Link>
            <Link to="/players">Players</Link>
            <Link to="/contact">Contact</Link>
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/fixtures" element={<Fixtures />} />
          <Route path="/previous-games" element={<PreviousGames />} />
          <Route path="/players" element={<Players />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
