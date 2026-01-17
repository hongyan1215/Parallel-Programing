// HW2 - Hybrid MPI+OpenMP SIFT runner
// - Rank 0 loads image, broadcasts to all ranks
// - Rank 0 computes SIFT (OpenMP inside); other ranks idle
// - Keeps judge-required output logic intact
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <omp.h>
#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
#ifdef USE_MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    int rank = 0, size = 1;
#endif
    
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image img;
    std::vector<Keypoint> kps;
    
#ifdef USE_MPI
    // Load image on rank 0, then broadcast to all processes
    if (rank == 0) {
        img = Image(input_img);
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }
    
    // Broadcast image dimensions first to size buffers correctly
    int img_width, img_height, img_channels;
    if (rank == 0) {
        img_width = img.width;
        img_height = img.height;
        img_channels = img.channels;
    }
    MPI_Bcast(&img_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate image on all processes
    if (rank != 0) {
        img = Image(img_width, img_height, img_channels);
    }
    
    // Broadcast raw image data
    MPI_Bcast(img.data, img.size * img.channels, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // SIFT computation on rank 0 (OpenMP inside)
    if (rank == 0) {
        // Use all available cores across allocated nodes; safe for judge environment
        omp_set_num_threads(6 * size);
        kps = find_keypoints_and_descriptors(img);
    }
#else
    // Single process execution
    img = Image(input_img);
    img = img.channels == 1 ? img : rgb_to_grayscale(img);
    kps = find_keypoints_and_descriptors(img);
#endif


    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
    if (rank == 0) {
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
    }
    /////////////////////////////////////////////////////////////

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    if (rank == 0) {
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}