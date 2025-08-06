import { storage } from '@forge/api';

/**
 * Store a job in Forge storage
 * @param {string} jobId - The job ID
 * @param {Object} jobData - The job data
 */
export const storeJob = async (jobId, jobData) => {
  try {
    await storage.set(`job_${jobId}`, jobData);
    console.log(`Job ${jobId} stored in Forge storage`);
  } catch (error) {
    console.error(`Error storing job ${jobId}:`, error);
    throw error;
  }
};

/**
 * Get a job from Forge storage
 * @param {string} jobId - The job ID
 * @returns {Object} - The job data
 */
export const getJob = async (jobId) => {
  try {
    const jobData = await storage.get(`job_${jobId}`);
    return jobData;
  } catch (error) {
    console.error(`Error getting job ${jobId}:`, error);
    throw error;
  }
};

/**
 * Update a job in Forge storage
 * @param {string} jobId - The job ID
 * @param {Object} jobData - The job data
 */
export const updateJob = async (jobId, jobData) => {
  try {
    await storage.set(`job_${jobId}`, jobData);
    console.log(`Job ${jobId} updated in Forge storage`);
  } catch (error) {
    console.error(`Error updating job ${jobId}:`, error);
    throw error;
  }
};

/**
 * Delete a job from Forge storage
 * @param {string} jobId - The job ID
 */
export const deleteJob = async (jobId) => {
  try {
    await storage.delete(`job_${jobId}`);
    console.log(`Job ${jobId} deleted from Forge storage`);
  } catch (error) {
    console.error(`Error deleting job ${jobId}:`, error);
    throw error;
  }
};

/**
 * List all jobs in Forge storage
 * @returns {Array} - Array of job IDs
 */
export const listJobs = async () => {
  try {
    const keys = await storage.query()
      .where('key', startsWith('job_'))
      .getKeys();
    
    return keys.map(key => key.replace('job_', ''));
  } catch (error) {
    console.error('Error listing jobs:', error);
    throw error;
  }
};

/**
 * Clean up old jobs from Forge storage
 * @param {number} maxJobs - Maximum number of jobs to keep
 */
export const cleanupOldJobs = async (maxJobs = 10) => {
  try {
    const jobs = await storage.query()
      .where('key', startsWith('job_'))
      .getMany();
    
    // Sort by start time (oldest first)
    jobs.sort((a, b) => a.value.startTime - b.value.startTime);
    
    // Delete old jobs if there are more than maxJobs
    if (jobs.length > maxJobs) {
      const jobsToDelete = jobs.slice(0, jobs.length - maxJobs);
      
      for (const job of jobsToDelete) {
        const jobId = job.key.replace('job_', '');
        await deleteJob(jobId);
      }
      
      console.log(`Cleaned up ${jobsToDelete.length} old jobs`);
    }
  } catch (error) {
    console.error('Error cleaning up old jobs:', error);
    // Don't throw the error, just log it
  }
};
