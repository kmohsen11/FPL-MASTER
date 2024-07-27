import { useState } from 'react'

import './App.css'
import Navbar from './components/Navbar'
import Contact from './components/contact'
import Home from './components/home'
import Fixtures from './components/fixtures'
import Stats from './components/players'

import { Route, Routes } from 'react-router-dom'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Navbar />
      <div className='container' >
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/contact' element={<Contact />} />
          <Route path='/fixtures' element={<Fixtures />} />
          <Route path='/players' element={<Stats />} />
        
     
        </Routes>
      </div>
    </>
  )
}

export default App
