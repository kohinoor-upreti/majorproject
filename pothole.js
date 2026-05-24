import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

import Navbar from './components/Navbar';
import Home from './components/Home';
import Login from './components/Login';
import Register from './components/Register';
import Prediction from './components/Prediction';
import VideoPrediction from './components/VideoPrediction';
import WebcamPrediction from './components/WebcamPrediction';
import History from './components/History';
import { getAuthToken } from './config/auth';

const ProtectedRoute = ({ children }) => {
  const token = getAuthToken();
  if (!token) return <Navigate to="/login" replace />;
  return children;
};

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/image-prediction" element={<ProtectedRoute><Prediction /></ProtectedRoute>} />
        <Route path="/video-prediction" element={<ProtectedRoute><VideoPrediction /></ProtectedRoute>} />
        <Route path="/webcam-prediction" element={<ProtectedRoute><WebcamPrediction /></ProtectedRoute>} />
        <Route path="/history" element={<ProtectedRoute><History /></ProtectedRoute>} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
      </Routes>
    </Router>
  );
}

export default App;
