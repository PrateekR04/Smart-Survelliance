import { useState, useEffect } from 'react'

const API_KEY = 'login-now'
const BASIC_AUTH = btoa(`guard:${API_KEY}`)

// API Helper
const api = {
    fetch: async (url, options = {}) => {
        const res = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY,
                'Authorization': `Basic ${BASIC_AUTH}`,
                ...options.headers,
            },
        })
        if (!res.ok) throw new Error(`API Error: ${res.status}`)
        if (res.status === 204) return null
        return res.json()
    },
}

// Login Component
function Login({ onLogin }) {
    const [username, setUsername] = useState('')
    const [role, setRole] = useState('guard')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (username.trim()) {
            onLogin({ username, role })
        }
    }

    return (
        <div className="login-container">
            <div className="login-card">
                <h1 className="login-title">🚗 Smart Parking</h1>
                <p className="login-subtitle">Guard Dashboard Login</p>

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="form-label">Username</label>
                        <input
                            type="text"
                            className="form-input"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter your name"
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Role</label>
                        <select
                            className="form-select"
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                        >
                            <option value="guard">Guard</option>
                            <option value="admin">Admin</option>
                        </select>
                    </div>

                    <button type="submit" className="btn btn-primary">
                        Login
                    </button>
                </form>
            </div>
        </div>
    )
}

// Guard Dashboard
function GuardDashboard({ user, onLogout }) {
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)

    const fetchAlerts = async () => {
        try {
            const data = await api.fetch('/api/v1/alerts')
            setAlerts(data.alerts || [])
        } catch (err) {
            console.error('Failed to fetch alerts:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchAlerts()
        const interval = setInterval(fetchAlerts, 10000) // Refresh every 10s
        return () => clearInterval(interval)
    }, [])

    const acknowledgeAlert = async (alertId) => {
        try {
            await api.fetch(`/api/v1/alerts/${alertId}/acknowledge`, { method: 'POST' })
            setAlerts(alerts.filter(a => a.id !== alertId))
        } catch (err) {
            console.error('Failed to acknowledge:', err)
        }
    }

    return (
        <div className="dashboard">
            <Sidebar user={user} onLogout={onLogout} activeTab="alerts" />

            <div className="main-content">
                <div className="page-header">
                    <h1 className="page-title">🚨 Active Alerts</h1>
                    <p className="page-subtitle">
                        {alerts.length} pending alerts • Auto-refreshing every 10s
                    </p>
                </div>

                {loading ? (
                    <div className="empty-state">Loading alerts...</div>
                ) : alerts.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">✅</div>
                        <p>No pending alerts. All clear!</p>
                    </div>
                ) : (
                    <div className="alerts-grid">
                        {alerts.map(alert => (
                            <div key={alert.id} className="alert-card unread">
                                {alert.image_path && (
                                    <img
                                        src={`/storage/images/${alert.image_path}`}
                                        alt="Vehicle"
                                        className="alert-image"
                                        onError={(e) => e.target.style.display = 'none'}
                                    />
                                )}
                                <div className="alert-content">
                                    <div className="alert-plate">
                                        {alert.plate_number || 'UNKNOWN'}
                                    </div>
                                    <div className="alert-meta">
                                        <span>📷 {alert.camera_id}</span>
                                        <span>🕐 {new Date(alert.timestamp).toLocaleTimeString()}</span>
                                    </div>
                                    <div className="alert-actions">
                                        <button
                                            className="btn btn-success btn-sm"
                                            onClick={() => acknowledgeAlert(alert.id)}
                                        >
                                            ✓ Acknowledge
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}

// Admin Dashboard
function AdminDashboard({ user, onLogout }) {
    const [activeTab, setActiveTab] = useState('logs')

    return (
        <div className="dashboard">
            <Sidebar
                user={user}
                onLogout={onLogout}
                activeTab={activeTab}
                onTabChange={setActiveTab}
                isAdmin
            />

            <div className="main-content">
                {activeTab === 'logs' && <LogsView />}
                {activeTab === 'whitelist' && <WhitelistView />}
            </div>
        </div>
    )
}

// Logs View
function LogsView() {
    const [logs, setLogs] = useState([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState('')

    useEffect(() => {
        fetchLogs()
    }, [filter])

    const fetchLogs = async () => {
        setLoading(true)
        try {
            const url = filter ? `/api/v1/logs?status=${filter}` : '/api/v1/logs'
            const data = await api.fetch(url)
            setLogs(data.logs || [])
        } catch (err) {
            console.error('Failed to fetch logs:', err)
        } finally {
            setLoading(false)
        }
    }

    const getStatusBadge = (status) => {
        const badges = {
            authorized: 'badge-success',
            unauthorized: 'badge-danger',
            unknown: 'badge-warning',
        }
        return badges[status] || 'badge-warning'
    }

    return (
        <>
            <div className="page-header">
                <h1 className="page-title">📋 Access Logs</h1>
                <p className="page-subtitle">View all vehicle access attempts</p>
            </div>

            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recent Logs</span>
                    <select
                        className="form-select"
                        style={{ width: 'auto' }}
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    >
                        <option value="">All Status</option>
                        <option value="authorized">Authorized</option>
                        <option value="unauthorized">Unauthorized</option>
                        <option value="unknown">Unknown</option>
                    </select>
                </div>

                {loading ? (
                    <div className="card-body">Loading...</div>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Plate</th>
                                <th>Camera</th>
                                <th>Status</th>
                                <th>Confidence</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {logs.map(log => (
                                <tr key={log.id}>
                                    <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                        {log.plate_number || '—'}
                                    </td>
                                    <td>{log.camera_id}</td>
                                    <td>
                                        <span className={`badge ${getStatusBadge(log.status)}`}>
                                            {log.status}
                                        </span>
                                    </td>
                                    <td>{(log.confidence * 100).toFixed(1)}%</td>
                                    <td>{new Date(log.timestamp).toLocaleString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </>
    )
}

// Whitelist View
function WhitelistView() {
    const [entries, setEntries] = useState([])
    const [loading, setLoading] = useState(true)
    const [showModal, setShowModal] = useState(false)
    const [formData, setFormData] = useState({
        plate_number: '',
        owner_name: '',
        vehicle_type: 'car',
    })

    useEffect(() => {
        fetchEntries()
    }, [])

    const fetchEntries = async () => {
        try {
            const data = await api.fetch('/api/v1/whitelist')
            setEntries(data.entries || [])
        } catch (err) {
            console.error('Failed to fetch whitelist:', err)
        } finally {
            setLoading(false)
        }
    }

    const addEntry = async (e) => {
        e.preventDefault()
        try {
            await api.fetch('/api/v1/whitelist', {
                method: 'POST',
                body: JSON.stringify(formData),
            })
            setShowModal(false)
            setFormData({ plate_number: '', owner_name: '', vehicle_type: 'car' })
            fetchEntries()
        } catch (err) {
            alert('Failed to add entry. Plate may already exist.')
        }
    }

    const deleteEntry = async (id) => {
        if (!confirm('Remove this plate from whitelist?')) return
        try {
            await api.fetch(`/api/v1/whitelist/${id}`, { method: 'DELETE' })
            fetchEntries()
        } catch (err) {
            console.error('Failed to delete:', err)
        }
    }

    return (
        <>
            <div className="page-header">
                <h1 className="page-title">✅ Whitelist</h1>
                <p className="page-subtitle">Manage authorized vehicles</p>
            </div>

            <div className="card">
                <div className="card-header">
                    <span className="card-title">{entries.length} Authorized Plates</span>
                    <button className="btn btn-primary btn-sm" onClick={() => setShowModal(true)}>
                        + Add Plate
                    </button>
                </div>

                {loading ? (
                    <div className="card-body">Loading...</div>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Plate Number</th>
                                <th>Owner</th>
                                <th>Vehicle Type</th>
                                <th>Added</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {entries.map(entry => (
                                <tr key={entry.id}>
                                    <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                        {entry.plate_number}
                                    </td>
                                    <td>{entry.owner_name}</td>
                                    <td>{entry.vehicle_type}</td>
                                    <td>{new Date(entry.created_at).toLocaleDateString()}</td>
                                    <td>
                                        <button
                                            className="btn btn-danger btn-sm"
                                            onClick={() => deleteEntry(entry.id)}
                                        >
                                            Remove
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <span>Add to Whitelist</span>
                            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
                        </div>
                        <form onSubmit={addEntry}>
                            <div className="modal-body">
                                <div className="form-group">
                                    <label className="form-label">Plate Number</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        value={formData.plate_number}
                                        onChange={e => setFormData({ ...formData, plate_number: e.target.value })}
                                        placeholder="MH12AB1234"
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Owner Name</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        value={formData.owner_name}
                                        onChange={e => setFormData({ ...formData, owner_name: e.target.value })}
                                        placeholder="John Doe"
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Vehicle Type</label>
                                    <select
                                        className="form-select"
                                        value={formData.vehicle_type}
                                        onChange={e => setFormData({ ...formData, vehicle_type: e.target.value })}
                                    >
                                        <option value="car">Car</option>
                                        <option value="motorcycle">Motorcycle</option>
                                        <option value="truck">Truck</option>
                                    </select>
                                </div>
                            </div>
                            <div className="modal-footer">
                                <button type="button" className="btn" onClick={() => setShowModal(false)}>
                                    Cancel
                                </button>
                                <button type="submit" className="btn btn-primary">
                                    Add Plate
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </>
    )
}

// Sidebar Component
function Sidebar({ user, onLogout, activeTab, onTabChange, isAdmin }) {
    return (
        <div className="sidebar">
            <div className="sidebar-logo">🚗 Smart Parking</div>
            <div className="sidebar-role">{user.role.toUpperCase()}</div>

            {isAdmin ? (
                <>
                    <div
                        className={`nav-item ${activeTab === 'logs' ? 'active' : ''}`}
                        onClick={() => onTabChange('logs')}
                    >
                        📋 Access Logs
                    </div>
                    <div
                        className={`nav-item ${activeTab === 'whitelist' ? 'active' : ''}`}
                        onClick={() => onTabChange('whitelist')}
                    >
                        ✅ Whitelist
                    </div>
                </>
            ) : (
                <div className="nav-item active">🚨 Alerts</div>
            )}

            <div className="sidebar-footer">
                <div style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '12px' }}>
                    👤 {user.username}
                </div>
                <button className="logout-btn" onClick={onLogout}>
                    Logout
                </button>
            </div>
        </div>
    )
}

// Main App
function App() {
    const [user, setUser] = useState(() => {
        const saved = localStorage.getItem('user')
        return saved ? JSON.parse(saved) : null
    })

    const handleLogin = (userData) => {
        setUser(userData)
        localStorage.setItem('user', JSON.stringify(userData))
    }

    const handleLogout = () => {
        setUser(null)
        localStorage.removeItem('user')
    }

    if (!user) {
        return <Login onLogin={handleLogin} />
    }

    if (user.role === 'admin') {
        return <AdminDashboard user={user} onLogout={handleLogout} />
    }

    return <GuardDashboard user={user} onLogout={handleLogout} />
}

export default App
