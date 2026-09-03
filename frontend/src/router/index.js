import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { keepAlive: true }
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('../views/ChatView.vue'),
      meta: { keepAlive: true }
    },
    {
      path: '/kg',
      name: 'knowledge-graph',
      component: () => import('../views/GraphView.vue'),
      meta: { keepAlive: true }
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
      meta: { keepAlive: true }
    }
  ]
})

export default router
