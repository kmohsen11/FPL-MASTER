import { useEffect, useState, useMemo } from 'react';
import "./Fixtures_style.css";

const API_BASE_URL = "https://fpl-master-48c1932d5d3b.herokuapp.com/api"; // No need for a proxy

// ✅ Hardcoded team logos based on FPL team IDs
const teamLogos = {
  1: { name: "Arsenal", logo: "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg" },
  2: { name: "Aston Villa", logo: "https://upload.wikimedia.org/wikipedia/en/9/9a/Aston_Villa_FC_new_crest.svg" },
  3: { name: "Brentford", logo: "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg" },
  4: { name: "Brighton", logo: "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg" },
  5: { name: "Chelsea", logo: "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg" },
  6: { name: "Crystal Palace", logo: "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg" },
  7: { name: "Everton", logo: "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg" },
  8: { name: "Liverpool", logo: "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg" },
  9: { name: "Man City", logo: "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg" },
  10: { name: "Man Utd", logo: "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg" },
  11: { name: "Newcastle", logo: "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg" },
  12: { name: "Southampton", logo: "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg" },
  13: { name: "Spurs", logo: "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg" },
  14: { name: "West Ham", logo: "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg" },
  15: { name: "Wolves", logo: "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg" },
  16: { name: "Nott'm Forest", logo: "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg" },
  17: { name: "Leicester", logo: "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg" },
  18: { name: "Fulham", logo: "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg" },
  19: { name: "Bournemouth", logo: "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg" },
  20: { name: "Watford", logo: "https://upload.wikimedia.org/wikipedia/en/e/e2/Watford.svg" },
  21: { name: "Ipswich", logo: "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg" },
};

function Fixtures() {
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // ✅ Fetch only fixtures (no need for /api/teams)
        const fixturesResponse = await fetch(`${API_BASE_URL}/fixtures`, {
          headers: { "Content-Type": "application/json" }
        });
        const fixturesData = await fixturesResponse.json();

        // ✅ Filter & map fixtures, adding hardcoded team logos
        const filteredFixtures = fixturesData.fixtures
          .filter(fixture => fixture.kickoff_time && new Date(fixture.kickoff_time) > new Date()) // Only upcoming fixtures
          .map(fixture => ({
            ...fixture,
            team_h: teamLogos[fixture.team_h] || { name: "Unknown", logo: "https://via.placeholder.com/40" },
            team_a: teamLogos[fixture.team_a] || { name: "Unknown", logo: "https://via.placeholder.com/40" },
          }));

        setFixtures(filteredFixtures);
      } catch (error) {
        console.error('Error fetching fixtures:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return loading ? (
    <div className="container">Loading fixtures...</div>
  ) : (
    <div className="container">
      <h2>Upcoming Premier League Fixtures</h2>
      {fixtures.length === 0 ? (
        <p>No fixtures available.</p>
      ) : (
        Object.entries(
          fixtures.reduce((acc, fixture) => {
            const { event } = fixture; // Game week number
            if (!acc[event]) acc[event] = [];
            acc[event].push(fixture);
            return acc;
          }, {})
        ).map(([gameWeek, fixtures]) => (
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
        ))
      )}
    </div>
  );
}

export default Fixtures;
