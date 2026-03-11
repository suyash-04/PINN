import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import PressureHead from './pages/PressureHead';
import FactorOfSafety from './pages/FactorOfSafety';
import ParameterExplorer from './pages/ParameterExplorer';
import HydrusComparison from './pages/HydrusComparison';
import SoilProperties from './pages/SoilProperties';
import ModelInfo from './pages/ModelInfo';
import Animation from './pages/Animation';
import TrainingLoss from './pages/TrainingLoss';
import PDEResidual from './pages/PDEResidual';
import ErrorAnalysis from './pages/ErrorAnalysis';
import Validation from './pages/Validation';
import Uncertainty from './pages/Uncertainty';
import CriticalSlip from './pages/CriticalSlip';
import RainfallSim from './pages/RainfallSim';
import ScenarioComparator from './pages/ScenarioComparator';
import Export from './pages/Export';

export default function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/pressure-head" element={<PressureHead />} />
            <Route path="/factor-of-safety" element={<FactorOfSafety />} />
            <Route path="/parameters" element={<ParameterExplorer />} />
            <Route path="/hydrus" element={<HydrusComparison />} />
            <Route path="/soil" element={<SoilProperties />} />
            <Route path="/model" element={<ModelInfo />} />
            <Route path="/animation" element={<Animation />} />
            <Route path="/training" element={<TrainingLoss />} />
            <Route path="/pde-residual" element={<PDEResidual />} />
            <Route path="/error" element={<ErrorAnalysis />} />
            <Route path="/validation" element={<Validation />} />
            <Route path="/uncertainty" element={<Uncertainty />} />
            <Route path="/critical-slip" element={<CriticalSlip />} />
            <Route path="/rainfall" element={<RainfallSim />} />
            <Route path="/scenarios" element={<ScenarioComparator />} />
            <Route path="/export" element={<Export />} />
          </Routes>
        </Layout>
      </AppProvider>
    </BrowserRouter>
  );
}
