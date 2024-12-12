import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './components/Home'
import Fixtures from './components/Fixtures'
import PreviousGames from './components/PreviousGames'
import Players from './components/Players'
import Contact from './components/Contact'
import './App.css'
import Navbar from './components/Navbar'

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
