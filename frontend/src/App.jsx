import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home_page';
import Fixtures from './components/Fixtures';
import PreviousGames from './components/PreviousGames_page';
import Players from './components/Players_page';
import Contact from './components/Contact_page';
import './App.css';
import Navbar from './components/Navbar_page';

function App() {
  return (
    <Router>
      <Navbar />
      <div>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/fixtures" element={<Fixtures />} />
          <Route path="/previous-games" element={<PreviousGames />} />
          <Route path="/players" element={<Players />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
