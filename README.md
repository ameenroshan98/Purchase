import { useEffect, useState } from "react";

function App() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);

  // Replace these with your own values
  const SHEET_ID = "YOUR_SPREADSHEET_ID";
  const API_KEY = "YOUR_API_KEY";
  const RANGE = "Sheet1!A1:D20";

  useEffect(() => {
    fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${RANGE}?key=${API_KEY}`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data.values) setRows(data.values);
        else setError("No data found in sheet");
      })
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>Google Sheets Dashboard</h1>
      {error ? (
        <p style={{ color: "red" }}>⚠️ Error: {error}</p>
      ) : (
        <table border="1" cellPadding="5">
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;
