// Basic frontend structure
// This is the homepage — will later connect to Spring Boot API to start scans

function App() {
  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h1>Cybersecurity Audit Tool</h1>
      <p>Enter your website URL to begin a scan.</p>
      <input type="text" placeholder="https://example.com" />
      <button>Start Scan</button>
    </div>
  );
}

export default App