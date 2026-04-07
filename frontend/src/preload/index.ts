import { contextBridge, ipcRenderer } from 'electron'
import * as fs from 'fs'

const api = {
  // File dialogs
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),
  saveFileDialog: (defaultName: string) => ipcRenderer.invoke('dialog:saveFile', defaultName),

  // Python RPC
  pythonCall: (method: string, params?: any) =>
    ipcRenderer.invoke('python:call', method, params || {}),
  pythonPing: () => ipcRenderer.invoke('python:ping'),

  // Progress listener
  onProgress: (callback: (data: { stage: string; pct: number; message: string }) => void) => {
    const handler = (_: any, data: any) => callback(data)
    ipcRenderer.on('pipeline:progress', handler)
    return () => ipcRenderer.removeListener('pipeline:progress', handler)
  },

  // File system (for reading binary transfer files)
  readBinaryFile: (path: string): ArrayBuffer => {
    const buffer = fs.readFileSync(path)
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  },
}

contextBridge.exposeInMainWorld('api', api)

export type ElectronAPI = typeof api
