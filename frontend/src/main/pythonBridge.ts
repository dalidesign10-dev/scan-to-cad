import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'
import { EventEmitter } from 'events'
import * as http from 'http'

/**
 * Spawns the FastAPI backend (backend/src/http_server.py) as a child
 * process of the Electron main process and waits until it's serving on
 * http://localhost:8321. The renderer talks to the backend over HTTP,
 * not IPC — this class just manages the subprocess lifecycle and
 * readiness check.
 *
 * The old JSON-RPC stdio plumbing was removed; it was dead code (no
 * component in the renderer ever called window.api.pythonCall).
 */
export class PythonBridge extends EventEmitter {
  private proc: ChildProcess | null = null
  private readonly port = 8321

  async spawn(): Promise<void> {
    // __dirname at runtime:
    //   dev mode:  frontend/out/main/
    //   packaged:  <asar>/out/main/
    // In both cases the backend lives three levels up at <repo>/backend/src.
    const projectRoot = join(__dirname, '..', '..', '..')
    const backendDir = join(projectRoot, 'backend', 'src')
    const scriptPath = join(backendDir, 'http_server.py')
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3'

    console.log('[PythonBridge] spawning', pythonPath, scriptPath)

    this.proc = spawn(pythonPath, [scriptPath], {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONPATH: backendDir,
        PYTHONUNBUFFERED: '1',
      },
    })

    this.proc.stdout?.on('data', (data: Buffer) => {
      process.stdout.write('[backend] ' + data.toString())
    })
    this.proc.stderr?.on('data', (data: Buffer) => {
      process.stderr.write('[backend] ' + data.toString())
    })
    this.proc.on('error', (err) => {
      console.error('[PythonBridge] spawn error:', err)
    })
    this.proc.on('exit', (code, signal) => {
      console.log(`[PythonBridge] backend exited code=${code} signal=${signal}`)
      this.proc = null
    })

    // Poll /docs (FastAPI auto-generated) until 200 OK or 30s timeout.
    const deadline = Date.now() + 30_000
    while (Date.now() < deadline) {
      if (this.proc == null) {
        throw new Error(
          'backend process exited before becoming ready — check [backend] output above',
        )
      }
      if (await this.probe()) {
        console.log('[PythonBridge] backend is up on http://localhost:' + this.port)
        return
      }
      await new Promise((r) => setTimeout(r, 300))
    }
    throw new Error(
      `backend did not come up on http://localhost:${this.port} within 30s`,
    )
  }

  private probe(): Promise<boolean> {
    return new Promise((resolve) => {
      const req = http.request(
        {
          host: '127.0.0.1',
          port: this.port,
          path: '/docs',
          method: 'GET',
          timeout: 1000,
        },
        (res) => {
          res.resume()
          resolve((res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 500)
        },
      )
      req.on('error', () => resolve(false))
      req.on('timeout', () => {
        req.destroy()
        resolve(false)
      })
      req.end()
    })
  }

  kill() {
    if (this.proc) {
      console.log('[PythonBridge] killing backend pid=' + this.proc.pid)
      this.proc.kill()
      this.proc = null
    }
  }
}
