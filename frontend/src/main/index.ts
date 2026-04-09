import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { PythonBridge } from './pythonBridge'
import { registerIpcHandlers } from './ipcHandlers'

let mainWindow: BrowserWindow | null = null
let pythonBridge: PythonBridge | null = null

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'Geomagic Claude — Scan to CAD',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // In dev mode, load from vite dev server
  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(async () => {
  // Open the main window immediately so the user sees something while
  // the Python backend is starting up. The renderer only talks to the
  // backend on explicit user actions (button clicks), so a brief window
  // where the UI is loaded but the backend isn't yet is fine.
  createWindow()

  // Start Python backend in parallel. If it fails, tell the user
  // exactly what's wrong instead of leaving a silent broken app.
  pythonBridge = new PythonBridge()
  try {
    await pythonBridge.spawn()
  } catch (err: any) {
    console.error('[main] backend failed to start:', err)
    dialog.showErrorBox(
      'Backend failed to start',
      [
        'The Python backend could not be launched.',
        '',
        err?.message ?? String(err),
        '',
        'Common causes:',
        '  • Python is not on PATH — install Python 3.10+',
        '  • Backend deps missing — run:',
        '      python -m pip install -r backend/requirements.txt',
        '  • Port 8321 is already in use by another process',
        '',
        'The app window will remain open but pipeline actions will fail',
        'until the backend is running.',
      ].join('\n'),
    )
  }

  registerIpcHandlers(pythonBridge)
})

app.on('window-all-closed', () => {
  pythonBridge?.kill()
  app.quit()
})

app.on('before-quit', () => {
  pythonBridge?.kill()
})
