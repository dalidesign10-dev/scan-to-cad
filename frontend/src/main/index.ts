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
  // Start Python backend
  pythonBridge = new PythonBridge()
  await pythonBridge.spawn()

  // Register IPC handlers
  registerIpcHandlers(pythonBridge)

  createWindow()
})

app.on('window-all-closed', () => {
  pythonBridge?.kill()
  app.quit()
})

app.on('before-quit', () => {
  pythonBridge?.kill()
})
