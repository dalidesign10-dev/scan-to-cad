import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'
import { EventEmitter } from 'events'
import { app } from 'electron'

interface PendingRequest {
  resolve: (value: any) => void
  reject: (reason: any) => void
}

export class PythonBridge extends EventEmitter {
  private proc: ChildProcess | null = null
  private pending: Map<number, PendingRequest> = new Map()
  private nextId = 1
  private buffer = ''

  async spawn(): Promise<void> {
    // __dirname is frontend/out/main in dev, resolve to project root
    const projectRoot = join(__dirname, '..', '..', '..')
    const backendDir = join(projectRoot, 'backend', 'src')
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3'

    console.log('Backend dir:', backendDir)

    return new Promise((resolve, reject) => {
      this.proc = spawn(pythonPath, ['server.py'], {
        cwd: backendDir,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONPATH: backendDir,
          PYTHONUNBUFFERED: '1',
        },
      })

      this.proc.stdout?.on('data', (data: Buffer) => {
        this.buffer += data.toString()
        this.processBuffer()
      })

      this.proc.stderr?.on('data', (data: Buffer) => {
        const msg = data.toString()
        console.error('[Python]', msg)
        if (msg.includes('Python backend started')) {
          resolve()
        }
      })

      this.proc.on('error', (err) => {
        console.error('Failed to spawn Python:', err)
        reject(err)
      })

      this.proc.on('exit', (code) => {
        console.log('Python process exited with code', code)
        for (const [id, pending] of this.pending) {
          pending.reject(new Error('Python process exited'))
        }
        this.pending.clear()
      })

      // Resolve after timeout if stderr message not received
      setTimeout(() => resolve(), 2000)
    })
  }

  private processBuffer() {
    const lines = this.buffer.split('\n')
    this.buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const msg = JSON.parse(line)
        this.handleMessage(msg)
      } catch (e) {
        console.error('Failed to parse Python output:', line)
      }
    }
  }

  private handleMessage(msg: any) {
    // Progress notification (no id)
    if (msg.method === 'progress') {
      this.emit('progress', msg.params)
      return
    }

    // Response to a request
    if (msg.id != null) {
      const pending = this.pending.get(msg.id)
      if (pending) {
        this.pending.delete(msg.id)
        if (msg.error) {
          pending.reject(new Error(msg.error.message))
        } else {
          pending.resolve(msg.result)
        }
      }
    }
  }

  async call(method: string, params: Record<string, any> = {}): Promise<any> {
    if (!this.proc?.stdin) {
      throw new Error('Python process not running')
    }

    const id = this.nextId++
    const request = { jsonrpc: '2.0', method, params, id }

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const line = JSON.stringify(request) + '\n'
      this.proc!.stdin!.write(line)
    })
  }

  kill() {
    this.proc?.kill()
    this.proc = null
  }
}
