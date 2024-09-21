import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import RestoreImage from './components/RestoreImage';
import TrainModels from './components/TrainModels';

import DefaultLayout from './components/NewDesign/DefaultLayout';
import Home from './components/Home';
const App: React.FC = () => {
  return (
    <Routes>
      <Route
        path="/restoreimage"
        element={
          <DefaultLayout>
            <RestoreImage />
          </DefaultLayout>
        }
      />
      <Route
        path="/trainmodels"
        element={
          <DefaultLayout>
            <TrainModels />{' '}
          </DefaultLayout>
        }
      />
      <Route
        path="/"
        element={
          <DefaultLayout>
            <Home />{' '}
          </DefaultLayout>
        }
      />
    </Routes>
  );
};

export default App;
