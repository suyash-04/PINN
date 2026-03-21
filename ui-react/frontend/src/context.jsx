import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from './api';

const AppCtx = createContext(null);

export function AppProvider({ children }) {
  const [defaults, setDefaults] = useState(null);
  const [geo, setGeo]   = useState(null);
  const [norm, setNorm] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.defaults()
      .then(d => {
        setDefaults(d);
        setGeo({ ...d.geo });
        setNorm({ ...d.norm });
      })
      .catch(err => setError(err.message));
  }, []);

  const updateGeo = useCallback(
    (key, value) => setGeo(prev => ({ ...prev, [key]: value })),
    []
  );

  const resetGeo = useCallback(
    () => defaults && setGeo({ ...defaults.geo }),
    [defaults]
  );

  if (error) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh',
      background:'var(--bg)', color:'var(--danger)', fontFamily:'var(--font-mono)', padding:24 }}>
      <div>
        <div style={{ fontSize:12, opacity:.6, marginBottom:8 }}>BACKEND UNREACHABLE</div>
        <div style={{ fontSize:14 }}>{error}</div>
        <div style={{ fontSize:11, opacity:.5, marginTop:8 }}>
          Start FastAPI: <code>uvicorn ui-react.api.main:app --reload</code>
        </div>
      </div>
    </div>
  );

  if (!defaults) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh',
      background:'var(--bg)', color:'var(--muted)', fontFamily:'var(--font-mono)' }}>
      <div style={{ textAlign:'center' }}>
        <Loader />
        <div style={{ marginTop:12, fontSize:12 }}>Loading PINN model…</div>
      </div>
    </div>
  );

  return (
    <AppCtx.Provider value={{ defaults, geo, norm, setGeo, updateGeo, resetGeo }}>
      {children}
    </AppCtx.Provider>
  );
}

function Loader() {
  return (
    <div style={{ width:32, height:32, border:'2px solid var(--border2)',
      borderTopColor:'var(--accent)', borderRadius:'50%', animation:'spin 0.8s linear infinite',
      margin:'0 auto' }} />
  );
}

export const useApp = () => useContext(AppCtx);
