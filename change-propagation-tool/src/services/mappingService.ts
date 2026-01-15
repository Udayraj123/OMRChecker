/**
 * Service for loading and managing file mappings
 */

import type { FileMappingData, FileMapping } from '../types';

class MappingService {
  private mappingData: FileMappingData | null = null;

  async loadMappings(): Promise<FileMappingData> {
    try {
      // In development, load from the parent directory
      const response = await fetch('../FILE_MAPPING.json');
      if (!response.ok) {
        throw new Error('Failed to load FILE_MAPPING.json');
      }
      const data: FileMappingData = await response.json();
      this.mappingData = data;
      return data;
    } catch (error) {
      console.error('Error loading mappings:', error);
      // Return empty data structure as fallback
      return {
        version: '2.0',
        mappings: [],
        statistics: {
          total: 0,
          synced: 0,
          partial: 0,
          not_started: 0,
          phase1: 0,
          phase2: 0,
          future: 0,
        },
      };
    }
  }

  getMappings(): FileMapping[] {
    return this.mappingData?.mappings || [];
  }

  getStatistics() {
    return this.mappingData?.statistics;
  }

  getMappingByPythonFile(pythonFile: string): FileMapping | undefined {
    return this.mappingData?.mappings.find((m) => m.python === pythonFile);
  }

  filterByStatus(status: string): FileMapping[] {
    return this.getMappings().filter((m) => m.status === status);
  }

  filterByPhase(phase: number | string): FileMapping[] {
    return this.getMappings().filter((m) => m.phase === phase);
  }

  filterByPriority(priority: string): FileMapping[] {
    return this.getMappings().filter((m) => m.priority === priority);
  }
}

export const mappingService = new MappingService();

