import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './components/home'
import Fixtures from './components/fixtures'
import PreviousGames from './components/previousGames'
import Players from './components/players'
import Contact from './components/contact'
import './App.css'
import Navbar from './components/navbar'

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
