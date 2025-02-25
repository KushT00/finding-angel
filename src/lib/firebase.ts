'use client';

import { getApp, getApps, initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyDpN7cbgxqfAncvdv8olmfpwq4OTv4Q5Jg",
  authDomain: "findmyangel-b281e.firebaseapp.com",
  projectId: "findmyangel-b281e",
  storageBucket: "findmyangel-b281e.firebasestorage.app",
  messagingSenderId: "750797441344",
  appId: "1:750797441344:web:d5191d7fd5f508fa5252a5",
  measurementId: "G-E5D6S3L5DS"
};

// Initialize Firebase
const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
const auth = getAuth(app);

export const db = getFirestore(app);

export { app, auth }; 