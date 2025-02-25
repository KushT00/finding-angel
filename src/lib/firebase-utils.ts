import { db } from './firebase';
import { 
  doc, 
  getDoc, 
  setDoc, 
  updateDoc,
  serverTimestamp 
} from 'firebase/firestore';

// Initialize user data when they first sign up
export const initializeUserData = async (uid: string, email: string, name: string) => {
  try {
    const userRef = doc(db, 'users', uid);
    const userDoc = await getDoc(userRef);

    if (!userDoc.exists()) {
      await setDoc(userRef, {
        email,
        name,
        credits: 5,
        viewed_profiles: [],
        created_at: serverTimestamp()
      });
    }
    return true;
  } catch (error) {
    console.error('Error initializing user data:', error);
    return false;
  }
};

export const getUserData = async (uid: string) => {
  try {
    const userRef = doc(db, 'users', uid);
    const userDoc = await getDoc(userRef);

    if (!userDoc.exists()) {
      return null;
    }

    return userDoc.data();
  } catch (error) {
    console.error('Error getting user data:', error);
    throw error;
  }
};

export const updateViewedProfile = async (uid: string, investorId: string) => {
  try {
    const userRef = doc(db, 'users', uid);
    const userDoc = await getDoc(userRef);

    if (!userDoc.exists()) {
      throw new Error('User not found');
    }

    const userData = userDoc.data();
    const viewed_profiles = userData.viewed_profiles || [];

    if (!viewed_profiles.includes(investorId)) {
      await updateDoc(userRef, {
        credits: userData.credits - 1,
        viewed_profiles: [...viewed_profiles, investorId],
        last_updated: serverTimestamp()
      });
    }

    return {
      credits: userData.credits - 1,
      viewed_profiles: [...viewed_profiles, investorId]
    };
  } catch (error) {
    console.error('Error updating viewed profile:', error);
    throw error;
  }
}; 