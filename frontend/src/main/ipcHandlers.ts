import { ipcMain, dialog, BrowserWindow } from 'electron'
import { PythonBridge } from './pythonBridge'

export function registerIpcHandlers(python: PythonBridge) {
  // Forward progress events to renderer
  python.on('progress', (data) => {
    const win = BrowserWindow.getAllWindows()[0]
    if (win) {
      win.webContents.send('pipeline:progress', data)
    }
  })

  // File open dialog
  ipcMain.handle('dialog:openFile', async () => {
    const result = await dialog.showOpenDialog({
      filters: [
        { name: '3D Meshes', extensions: ['stl', 'ply', 'obj'] },
        { name: 'All Files', extensions: ['*'] },
      ],
      properties: ['openFile'],
    })
    return result.canceled ? null : result.filePaths[0]
  })

  // File save dialog
  ipcMain.handle('dialog:saveFile', async (_, defaultName: string) => {
    const result = await dialog.showSaveDialog({
      defaultPath: defaultName,
      filters: [
        { name: 'STEP Files', extensions: ['step', 'stp'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    })
    return result.canceled ? null : result.filePath
  })

  // Generic Python RPC call
  ipcMain.handle('python:call', async (_, method: string, params: any) => {
    return python.call(method, params)
  })

  // Ping to check Python is alive
  ipcMain.handle('python:ping', async () => {
    return python.call('ping')
  })
}
