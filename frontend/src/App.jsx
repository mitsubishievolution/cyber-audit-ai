// App.jsx
// This is the root React component.
// It currently just displays a simple message so you can confirm React runs correctly.
// connect it to the Spring Boot backend via REST APIs.

function App() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial' }}>
      <h1>Cybersecurity Audit Tool</h1>
      <p>This is the React frontend. You'll use this page to enter a website URL and view scan results.</p>
    </div>
  );
}

export default App;
