import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import About from './pages/About'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Landing from './pages/Landing'
import RunDetail from './pages/RunDetail'
import Workspace from './pages/Workspace'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/verify" element={<Workspace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/history/:id" element={<RunDetail />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<p className="p-8 text-slate-600">Page not found.</p>} />
      </Route>
    </Routes>
  )
}
