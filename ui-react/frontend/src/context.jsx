import { createContext, useContext, useState, useEffect } from 'react';
import { api } from './api';

const AppCtx = createContext(null);

export function AppProvider({ children }) {
  const [defaults, setDefaults] = useState(null);
  const [geo, setGeo] = useState(null);
  const [norm, setNorm] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.defaults().then((d) => {
      setDefaults(d);
      setGeo({ ...d.geo });
      setNorm({ ...d.norm });
      setLoading(false);
    });
  }, []);

  const updateGeo = (key, value) => setGeo((prev) => ({ ...prev, [key]: value }));

  if (loading) return (
    <div className="flex items-center justify-center h-screen bg-slate-50">
      <div className="text-center space-y-4">
        <div className="animate-spin h-12 w-12 border-4 border-blue-600 border-t-transparent rounded-full mx-auto" />
        <p className="text-slate-600 text-lg">Loading PINN model…</p>
      </div>
    </div>
  );

  return (
    <AppCtx.Provider value={{ defaults, geo, norm, setGeo, updateGeo }}>
      {children}
    </AppCtx.Provider>
  );
}

export const useApp = () => useContext(AppCtx);
