import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context';
import Layout from './components/Layout';
import { ErrorBoundary } from './components/ui';

import Overview           from './pages/Overview';
import PressureHead       from './pages/PressureHead';
import FactorOfSafety     from './pages/FactorOfSafety';
import ParameterExplorer  from './pages/ParameterExplorer';
import HydrusComparison   from './pages/HydrusComparison';
import SoilProperties     from './pages/SoilProperties';
import ModelInfo          from './pages/ModelInfo';
import Animation          from './pages/Animation';
import TrainingLoss       from './pages/TrainingLoss';
import PDEResidual        from './pages/PDEResidual';
import ErrorAnalysis      from './pages/ErrorAnalysis';
import Validation         from './pages/Validation';
import Uncertainty        from './pages/Uncertainty';
import CriticalSlip       from './pages/CriticalSlip';
import RainfallSim        from './pages/RainfallSim';
import ScenarioComparator from './pages/ScenarioComparator';
import Export             from './pages/Export';

const ROUTES = [
  { path: '/',                element: <Overview /> },
  { path: '/pressure-head',  element: <PressureHead /> },
  { path: '/factor-of-safety', element: <FactorOfSafety /> },
  { path: '/parameters',     element: <ParameterExplorer /> },
  { path: '/hydrus',         element: <HydrusComparison /> },
  { path: '/soil',           element: <SoilProperties /> },
  { path: '/model',          element: <ModelInfo /> },
  { path: '/animation',      element: <Animation /> },
  { path: '/training',       element: <TrainingLoss /> },
  { path: '/pde-residual',   element: <PDEResidual /> },
  { path: '/error',          element: <ErrorAnalysis /> },
  { path: '/validation',     element: <Validation /> },
  { path: '/uncertainty',    element: <Uncertainty /> },
  { path: '/critical-slip',  element: <CriticalSlip /> },
  { path: '/rainfall',       element: <RainfallSim /> },
  { path: '/scenarios',      element: <ScenarioComparator /> },
  { path: '/export',         element: <Export /> },
];

export default function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <Layout>
          <Routes>
            {ROUTES.map(({ path, element }) => (
              <Route
                key={path}
                path={path}
                end={path === '/'}
                element={<ErrorBoundary>{element}</ErrorBoundary>}
              />
            ))}
          </Routes>
        </Layout>
      </AppProvider>
    </BrowserRouter>
  );
}
