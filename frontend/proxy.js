import express from 'express'
import cors from 'cors'
import fetch from 'node-fetch' // Ensure this is installed

const app = express()
const PORT = 4000

app.use(cors())

app.get('/api/fixtures', async (req, res) => {
  try {
    const response = await fetch(
      'https://fantasy.premierleague.com/api/fixtures/'
    )
    const data = await response.json()
    res.json(data)
  } catch (error) {
    console.error('Error fetching fixtures from FPL API:', error)
    res.status(500).json({ error: 'Failed to fetch fixtures' })
  }
})

app.get('/api/teams', async (req, res) => {
  try {
    const response = await fetch(
      'https://fantasy.premierleague.com/api/bootstrap-static/'
    )
    const data = await response.json()
    res.json(data)
  } catch (error) {
    console.error('Error fetching teams from FPL API:', error)
    res.status(500).json({ error: 'Failed to fetch teams' })
  }
})

app.listen(PORT, () => {
  console.log(`Proxy server is running on http://localhost:${PORT}`)
})
