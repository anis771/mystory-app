import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HomePage from './components/Home';
import ExperimentsPage from './components/Experiments'; 
import JsonStoryViewer from './components/viewer'
const StoryGeneratorApp = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/experiments" element={<ExperimentsPage />} />
       <Route path="/viewer" element={<JsonStoryViewer />} />
      {/* You can add more routes here for other pages */}
    </Routes>
  );
};

export default StoryGeneratorApp;