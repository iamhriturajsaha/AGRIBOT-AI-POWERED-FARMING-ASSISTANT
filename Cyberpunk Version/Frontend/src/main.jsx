import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './index.css'
import { Toaster } from 'react-hot-toast'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
    <Toaster 
      position="top-right" 
      toastOptions={{
        style: {
          background: '#1A1A1D',
          color: '#fff',
          border: '1px solid rgba(255,255,255,0.1)',
        }
      }}
    />
  </React.StrictMode>,
)
