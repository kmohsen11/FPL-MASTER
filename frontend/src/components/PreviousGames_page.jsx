import { useEffect, useState } from 'react';
import "./PreviousGames_style.css";

const API_BASE_URL = "https://fpl-master-48c1932d5d3b.herokuapp.com/api"; // Direct Flask API

// ✅ Corrected team IDs with logos
const teamLogos = {
  1: { name: "Arsenal", logo: "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg" },
  2: { name: "Fulham", logo: "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg" },
  3: { name: "Aston Villa", logo: "https://upload.wikimedia.org/wikipedia/en/9/9a/Aston_Villa_FC_new_crest.svg" },
  4: { name: "Bournemouth", logo: "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg" },
  5: { name: "Brentford", logo: "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg" },
  6: { name: "Brighton", logo: "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg" },
  7: { name: "Leicester", logo: "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg" },
  8: { name: "Chelsea", logo: "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg" },
  9: { name: "Crystal Palace", logo: "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg" },
  10: { name: "Everton", logo: "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg" },
  11: { name: "Ipswich", logo: "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg" },
  12: { name: "Liverpool", logo: "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg" },
  13: { name: "Man City", logo: "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg" },
  14: { name: "Man Utd", logo: "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg" },
  15: { name: "Newcastle", logo: "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg" },
  16: { name: "Nott'm Forest", logo: "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg" },
  17: { name: "Southampton", logo: "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg" },
  18: { name: "Spurs", logo: "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg" },
  19: { name: "West Ham", logo: "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg" },
  20: { name: "Wolves", logo: "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg" },
};

function PreviousGames() {
  const [previousGames, setPreviousGames] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // ✅ Fetch fixtures from Flask API
        const fixturesResponse = await fetch(`${API_BASE_URL}/fixtures`, {
          headers: { "Content-Type": "application/json" },
        });
        const fixturesData = await fixturesResponse.json();

        // ✅ Debug API Response
        console.log("🚀 API Response:", fixturesData);

        if (!fixturesData.fixtures) {
          console.error("❌ Unexpected API response format!", fixturesData);
          return;
        }

        // ✅ Filter finished games only
        const finishedGames = fixturesData.fixtures
          .filter(game => game.finished) // Only finished games
          .map(game => ({
            ...game,
            team_h: teamLogos[game.team_h] || { name: "Unknown", logo: "https://via.placeholder.com/40" },
            team_a: teamLogos[game.team_a] || { name: "Unknown", logo: "https://via.placeholder.com/40" },
            team_h_score: game.team_h_score ?? 'N/A',
            team_a_score: game.team_a_score ?? 'N/A',
          }));

        console.log("✅ Filtered & Mapped Games:", finishedGames);

        setPreviousGames(finishedGames);
      } catch (error) {
        console.error('❌ Error fetching previous games:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // ✅ Group games by game week
  const groupedGames = previousGames.reduce((acc, game) => {
    const { event } = game;
    if (!acc[event]) {
      acc[event] = [];
    }
    acc[event].push(game);
    return acc;
  }, {});

  // ✅ Sort game weeks from newest to oldest (descending)
  const sortedGameWeeks = Object.entries(groupedGames).sort(
    (a, b) => parseInt(b[0]) - parseInt(a[0])
  );

  return loading ? (
    <div className="container">Loading previous games...</div>
  ) : (
    <div className="container">
      <h2>Previous Premier League Games</h2>
      {previousGames.length === 0 ? (
        <p>No previous games available.</p>
      ) : (
        sortedGameWeeks.map(([gameWeek, games]) => (
          <div key={gameWeek}>
            <h3 className="game-week-header">Game Week {gameWeek}</h3>
            <div className="games-grid">
              {games.map((game) => (
                <div className="game-card" key={game.id}>
                  <div className="team">
                    <img src={game.team_h.logo} alt={game.team_h.name} />
                    <span className="team-name">{game.team_h.name}</span>
                    <span className="team-score">({game.team_h_score})</span>
                  </div>
                  <div className="vs">VS</div>
                  <div className="team">
                    <img src={game.team_a.logo} alt={game.team_a.name} />
                    <span className="team-name">{game.team_a.name}</span>
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
        ))
      )}
    </div>
  );
}

export default PreviousGames;
