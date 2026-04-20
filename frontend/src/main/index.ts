import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { spawn, ChildProcess } from 'child_process'
import { PythonBridge } from './pythonBridge'
import { registerIpcHandlers } from './ipcHandlers'

let mainWindow: BrowserWindow | null = null
let pythonBridge: PythonBridge | null = null
let httpServerProc: ChildProcess | null = null

function spawnHttpServer() {
  const projectRoot = join(__dirname, '..', '..', '..')
  const backendDir = join(projectRoot, 'backend', 'src')
  const pythonPath = process.platform === 'win32' ? 'python' : 'python3'

  httpServerProc = spawn(pythonPath, ['http_server.py'], {
    cwd: backendDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONPATH: backendDir, PYTHONUNBUFFERED: '1' },
  })

  httpServerProc.stdout?.on('data', (d: Buffer) => console.log('[HTTP]', d.toString().trim()))
  httpServerProc.stderr?.on('data', (d: Buffer) => console.error('[HTTP]', d.toString().trim()))
  httpServerProc.on('exit', (code) => console.log('HTTP server exited with code', code))
}

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
  // Start Python backends
  pythonBridge = new PythonBridge()
  await pythonBridge.spawn()

  spawnHttpServer()

  // Register IPC handlers
  registerIpcHandlers(pythonBridge)

  createWindow()
})

app.on('window-all-closed', () => {
  pythonBridge?.kill()
  httpServerProc?.kill()
  app.quit()
})

app.on('before-quit', () => {
  pythonBridge?.kill()
  httpServerProc?.kill()
})
