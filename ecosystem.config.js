module.exports = {
  apps: [
    {
      name: 'wwai-macro-landing',
      cwd: '/mnt/nas/WWAI/WWAI-MACRO/wwai-macro-landing',
      script: 'npm',
      args: 'start',
      env: {
        NODE_ENV: 'production',
        PORT: 3801
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      log_file: '/tmp/wwai-macro-landing.log',
      error_file: '/tmp/wwai-macro-landing-error.log',
      out_file: '/tmp/wwai-macro-landing-out.log'
    },
    {
      name: 'wwai-gnn-frontend',
      cwd: '/mnt/nas/WWAI/WWAI-MACRO/WWAI-GNN/frontend',
      script: 'npm',
      args: 'start',
      env: {
        NODE_ENV: 'production',
        PORT: 3789
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      log_file: '/tmp/wwai-gnn-frontend.log',
      error_file: '/tmp/wwai-gnn-frontend-error.log',
      out_file: '/tmp/wwai-gnn-frontend-out.log'
    },
    {
      name: 'wwai-gnn-api',
      cwd: '/mnt/nas/WWAI/WWAI-MACRO/WWAI-GNN/api',
      script: 'main.py',
      interpreter: '/home/chae/miniconda3/bin/python3',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      log_file: '/tmp/wwai-gnn-api.log',
      error_file: '/tmp/wwai-gnn-api-error.log',
      out_file: '/tmp/wwai-gnn-api-out.log'
    }
  ]
}
