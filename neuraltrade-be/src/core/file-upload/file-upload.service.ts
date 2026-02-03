import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';

/**
 * File Upload Service for NeuralTrade
 * 
 * Handles:
 * - Profile photos
 * - Documents (KYC)
 * - Strategy exports
 * 
 * Note: For production, integrate with S3/CloudFlare R2
 */
@Injectable()
export class FileUploadService {
    private readonly logger = new Logger(FileUploadService.name);
    private readonly uploadDir: string;

    constructor() {
        this.uploadDir = process.env.UPLOAD_DIR || './uploads';
        this.ensureUploadDir();
    }

    private ensureUploadDir() {
        const dirs = ['profiles', 'documents', 'exports'];

        for (const dir of dirs) {
            const fullPath = path.join(this.uploadDir, dir);
            if (!fs.existsSync(fullPath)) {
                fs.mkdirSync(fullPath, { recursive: true });
                this.logger.log(`Created upload directory: ${fullPath}`);
            }
        }
    }

    /**
     * Upload a file to local storage
     */
    async uploadFile(
        buffer: Buffer,
        originalName: string,
        folder: 'profiles' | 'documents' | 'exports',
    ): Promise<{ fileName: string; filePath: string; url: string }> {
        const ext = path.extname(originalName);
        const fileName = `${uuidv4()}${ext}`;
        const filePath = path.join(this.uploadDir, folder, fileName);

        await fs.promises.writeFile(filePath, buffer);

        this.logger.log(`File uploaded: ${fileName} to ${folder}`);

        return {
            fileName,
            filePath,
            url: `/uploads/${folder}/${fileName}`,
        };
    }

    /**
     * Delete a file
     */
    async deleteFile(filePath: string): Promise<boolean> {
        try {
            const fullPath = path.resolve(filePath);

            if (!fullPath.startsWith(path.resolve(this.uploadDir))) {
                this.logger.warn('Attempted to delete file outside upload directory');
                return false;
            }

            await fs.promises.unlink(fullPath);
            this.logger.log(`File deleted: ${filePath}`);
            return true;
        } catch (error) {
            this.logger.error(`Failed to delete file: ${filePath}`, error);
            return false;
        }
    }

    /**
     * Get file stats
     */
    async getFileInfo(filePath: string): Promise<{
        exists: boolean;
        size?: number;
        createdAt?: Date;
    }> {
        try {
            const stats = await fs.promises.stat(filePath);
            return {
                exists: true,
                size: stats.size,
                createdAt: stats.birthtime,
            };
        } catch {
            return { exists: false };
        }
    }

    /**
     * Validate file type
     */
    validateFileType(
        mimetype: string,
        allowedTypes: string[],
    ): boolean {
        return allowedTypes.includes(mimetype);
    }

    /**
     * Validate file size
     */
    validateFileSize(size: number, maxSizeBytes: number): boolean {
        return size <= maxSizeBytes;
    }

    // Common validators
    isValidProfilePhoto(mimetype: string, size: number): boolean {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const maxSize = 5 * 1024 * 1024; // 5MB
        return this.validateFileType(mimetype, allowedTypes) && this.validateFileSize(size, maxSize);
    }

    isValidDocument(mimetype: string, size: number): boolean {
        const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png'];
        const maxSize = 10 * 1024 * 1024; // 10MB
        return this.validateFileType(mimetype, allowedTypes) && this.validateFileSize(size, maxSize);
    }
}
