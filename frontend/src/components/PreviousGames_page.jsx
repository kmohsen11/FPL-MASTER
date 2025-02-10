import { useEffect, useState } from 'react'
import "./PreviousGames_style.css";

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
  Everton: 'https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg',
  Liverpool: 'https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg',
  'Man City':
    'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Manchester_City_FC_badge.svg/380px-Manchester_City_FC_badge.svg.png',
  'Man Utd':
    'https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Manchester_United_FC_crest.svg/400px-Manchester_United_FC_crest.svg.png',
  Newcastle:
    'https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Newcastle_United_Logo.svg/400px-Newcastle_United_Logo.svg.png',
  Southampton:
    'https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg',
  Spurs: 'https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg',
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

function PreviousGames() {
  const [previousGames, setPreviousGames] = useState([])
  const [loading, setLoading] = useState(true)

  // Fetch and normalize data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch fixtures
        const fixturesResponse = await fetch(
          'http://localhost:4000/api/fixtures'
        )
        const fixturesData = await fixturesResponse.json()

        // Fetch teams
        const teamsResponse = await fetch('http://localhost:4000/api/teams')
        const teamsData = await teamsResponse.json()

        // Create a team mapping for logos and names
        const mapping = {}
        teamsData.teams.forEach((team) => {
          mapping[team.id] = {
            name: team.name,
            logo: teamLogos[team.name] || 'https://via.placeholder.com/40',
          }
        })
        // Filter finished games
        const finishedGames = fixturesData.filter((game) => game.finished)

        // Map game data
        const mappedGames = finishedGames.map((game) => ({
          id: game.id,
          event: game.event, // Game week number
          kickoff_time: game.kickoff_time,
          team_h: mapping[game.team_h],
          team_a: mapping[game.team_a],
          team_h_score: game.team_h_score ?? 'N/A',
          team_a_score: game.team_a_score ?? 'N/A',
        }))

        setPreviousGames(mappedGames)
      } catch (error) {
        console.error('Error fetching data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  // Group games by game week
  const groupedGames = previousGames.reduce((acc, game) => {
    const { event } = game
    if (!acc[event]) {
      acc[event] = []
    }
    acc[event].push(game)
    return acc
  }, {})

  // Sort game weeks from newest to oldest (descending)
  const sortedGameWeeks = Object.entries(groupedGames).sort(
    (a, b) => parseInt(b[0]) - parseInt(a[0])
  )

  if (loading) {
    return <div className="container">Loading previous games...</div>
  }

  if (!previousGames || previousGames.length === 0) {
    return (
      <div className="container">
        <h2>Previous Premier League Games</h2>
        <p>No previous games available.</p>
      </div>
    )
  }

  return (
    <div className="container">
      <h2>Previous Premier League Games</h2>
      {sortedGameWeeks.map(([gameWeek, games]) => (
        <div key={gameWeek}>
          <h3 className="game-week-header">Game Week {gameWeek}</h3>
          <div className="games-grid">
            {games.map((game) => (
              <div className="game-card" key={game.id}>
                <div className="team">
                  <img
                    src={game.team_h?.logo}
                    alt={game.team_h?.name || 'Unknown Team'}
                  />
                  <span className="team-name">
                    {game.team_h?.name || 'N/A'}
                  </span>
                  <span className="team-score">({game.team_h_score})</span>
                </div>
                <div className="vs">VS</div>
                <div className="team">
                  <img
                    src={game.team_a?.logo}
                    alt={game.team_a?.name || 'Unknown Team'}
                  />
                  <span className="team-name">
                    {game.team_a?.name || 'N/A'}
                  </span>
                  <span className="team-score">({game.team_a_score})</span>
                </div>
                <p className="kickoff-time">
                  {game.kickoff_time
                    ? new Date(game.kickoff_time).toLocaleString()
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

export default PreviousGames
