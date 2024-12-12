import { useEffect, useState } from 'react'
import './Fixtures.css'

function Fixtures() {
  const [fixtures, setFixtures] = useState([])
  const [loading, setLoading] = useState(true)

  const teamLogos = {
    Arsenal: 'https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg',
    'Aston Villa':
      'https://upload.wikimedia.org/wikipedia/en/thumb/9/9a/Aston_Villa_FC_new_crest.svg/300px-Aston_Villa_FC_new_crest.svg.png',
    Brentford:
      'https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg',
    Brighton:
      'https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg',
    Chelsea: 'https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg',
    'Crystal Palace':
      'https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/Crystal_Palace_FC_logo_%282022%29.svg/350px-Crystal_Palace_FC_logo_%282022%29.svg.png',
    Everton:
      'https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg',
    Liverpool:
      'https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg',
    'Man City':
      'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Manchester_City_FC_badge.svg/380px-Manchester_City_FC_badge.svg.png',
    'Man Utd':
      'https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Manchester_United_FC_crest.svg/400px-Manchester_United_FC_crest.svg.png',
    Newcastle:
      'https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Newcastle_United_Logo.svg/400px-Newcastle_United_Logo.svg.png',
    Southampton:
      'https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg',
    'Tottenham Hotspur':
      'https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg',
    'West Ham':
      'https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg',
    Wolves:
      'https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg',
    "Nott'm Forest":
      'https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/Nottingham_Forest_F.C._logo.svg/220px-Nottingham_Forest_F.C._logo.svg.png',
    Leicester:
      'https://upload.wikimedia.org/wikipedia/en/thumb/2/2d/Leicester_City_crest.svg/380px-Leicester_City_crest.svg.png',
    Fulham:
      'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Fulham_FC_%28shield%29.svg/300px-Fulham_FC_%28shield%29.svg.png',
    Bournemouth:
      'https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg',
    Watford: 'https://upload.wikimedia.org/wikipedia/en/e/e2/Watford.svg',
    Ipswich:
      'https://upload.wikimedia.org/wikipedia/en/thumb/4/43/Ipswich_Town.svg/312px-Ipswich_Town.svg.png',
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        const fixturesResponse = await fetch(
          'http://localhost:4000/api/fixtures'
        )
        const fixturesData = await fixturesResponse.json()

        const teamsResponse = await fetch('http://localhost:4000/api/teams')
        const teamsData = await teamsResponse.json()

        const teamMapping = {}
        teamsData.teams.forEach((team) => {
          teamMapping[team.id] = {
            name: team.name,
            logo: teamLogos[team.name] || 'https://via.placeholder.com/40',
          }
        })

        const filteredFixtures = fixturesData.filter((fixture) => {
          const fixtureDate = new Date(fixture.kickoff_time)
          return fixtureDate > new Date()
        })

        const mappedFixtures = filteredFixtures.map((fixture) => ({
          ...fixture,
          team_h: teamMapping[fixture.team_h],
          team_a: teamMapping[fixture.team_a],
        }))

        setFixtures(mappedFixtures)
      } catch (error) {
        console.error('Error fetching fixtures:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  // Group fixtures by game week
  const groupedFixtures = fixtures.reduce((acc, fixture) => {
    const { event } = fixture // Game week number
    if (!acc[event]) {
      acc[event] = []
    }
    acc[event].push(fixture)
    return acc
  }, {})

  if (loading) {
    return <div className="container">Loading fixtures...</div>
  }

  if (!fixtures || fixtures.length === 0) {
    return (
      <div className="container">
        <h2>Upcoming Premier League Fixtures</h2>
        <p>No fixtures available.</p>
      </div>
    )
  }

  return (
    <div className="container">
      <h2>Upcoming Premier League Fixtures</h2>
      {Object.entries(groupedFixtures).map(([gameWeek, fixtures]) => (
        <div key={gameWeek}>
          <h3 className="game-week-header">Game Week {gameWeek}</h3>
          <div className="fixtures-grid">
            {fixtures.map((fixture) => (
              <div className="fixture-card" key={fixture.id}>
                <div className="team">
                  <img src={fixture.team_h.logo} alt={fixture.team_h.name} />
                  <span>{fixture.team_h.name}</span>
                </div>
                <div className="vs">VS</div>
                <div className="team">
                  <img src={fixture.team_a.logo} alt={fixture.team_a.name} />
                  <span>{fixture.team_a.name}</span>
                </div>
                <p className="kickoff-time">
                  {fixture.kickoff_time
                    ? new Date(fixture.kickoff_time).toLocaleString()
                    : 'Kickoff time TBD'}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default Fixtures
