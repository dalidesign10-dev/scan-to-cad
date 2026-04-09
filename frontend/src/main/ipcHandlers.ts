import { ipcMain, dialog } from 'electron'
import { PythonBridge } from './pythonBridge'

/**
 * IPC handlers for the Electron main process. Only file dialogs are
 * live — the renderer talks to the Python backend over HTTP on
 * localhost:8321, not through IPC, so the old python:call / python:ping
 * handlers were removed along with the stdio JSON-RPC plumbing.
 */
export function registerIpcHandlers(_python: PythonBridge) {
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
}
