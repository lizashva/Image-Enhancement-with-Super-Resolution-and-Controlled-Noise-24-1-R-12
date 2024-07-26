import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import RestoreImage from "./components/RestoreImage";
import TrainModels from "./components/TrainModels";
import Metrics from "./components/matrics";

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/restoreimage" element={<RestoreImage />} />
          <Route path="/trainmodels" element={<TrainModels />} />
          <Route
            path="/metrics"
            element={
              <Metrics
                psnrData={{ train: [], val: [] }}
                ssimData={{ train: [], val: [] }}
              />
            }
          />
          <Route
            path="/"
            element={<div className="p-4">Welcome to the React App!</div>}
          />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
