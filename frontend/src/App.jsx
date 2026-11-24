import { useEffect, useState } from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { getToken, setToken } from './api';
import Login from './pages/Login.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Visitors from './pages/Visitors.jsx';
import VisitorDetail from './pages/VisitorDetail.jsx';
import Visits from './pages/Visits.jsx';
import Gate from './pages/Gate.jsx';

export default function App() {
  const [token, setTok] = useState(getToken());

  useEffect(() => {
    const onLogout = () => setTok(null);
    window.addEventListener('vt:logout', onLogout);
    return () => window.removeEventListener('vt:logout', onLogout);
  }, []);

  const login = (t) => {
    setToken(t);
    setTok(t);
  };
  const logout = () => {
    setToken(null);
    setTok(null);
  };

  return (
    <Routes>
      {/* The gate kiosk uses a gate key, not an admin login. */}
      <Route path="/gate" element={<Gate />} />
      <Route path="/login" element={token ? <Navigate to="/" /> : <Login onLogin={login} />} />
      <Route
        path="/*"
        element={
          token ? (
            <div className="layout">
              <nav className="sidebar">
                <div className="brand">VisiTrack</div>
                <NavLink to="/" end>Dashboard</NavLink>
                <NavLink to="/visitors">Visitors</NavLink>
                <NavLink to="/visits">Visit log</NavLink>
                <a href="/gate" target="_blank" rel="noreferrer">Open gate kiosk ↗</a>
                <button className="link" onClick={logout}>Log out</button>
              </nav>
              <main className="content">
                <Routes>
                  <Route path="/" element={<Dashboard token={token} />} />
                  <Route path="/visitors" element={<Visitors />} />
                  <Route path="/visitors/:id" element={<VisitorDetail />} />
                  <Route path="/visits" element={<Visits />} />
                  <Route path="*" element={<Navigate to="/" />} />
                </Routes>
              </main>
            </div>
          ) : (
            <Navigate to="/login" />
          )
        }
      />
    </Routes>
  );
}
